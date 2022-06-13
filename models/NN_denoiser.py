import faiss
import numpy as np
import torch
import torchvision

from utils import patch_to_window_index


class windowed_prior_wrapper:
    def __init__(self, priors, patch_indices):
        self.priors = priors
        self.patch_indices = patch_indices

    def denoise(self, queries, noise_var):
        for i in range(len(self.priors)):
            queries[self.patch_indices == i] = self.priors[i].denoise(queries[self.patch_indices == i], None)
        return queries


class FaissNNModule:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        self.index = None

    def _get_index(self, n, d):
        return faiss.IndexIVFFlat(faiss.IndexFlat(d), d, int(np.sqrt(n)))
        # return faiss.IndexIVFPQ(faiss.IndexFlatL2(d), d, int(np.sqrt(n)), 8, 8)

    def set_index(self, data):
        import torchvision
        self.data = data
        resized_data = torchvision.transforms.Resize((4, 4))(data.reshape(-1, 1, 8, 8)).reshape(-1, 4 * 4)
        self.resized_data = np.ascontiguousarray(resized_data.numpy(), dtype='float32')
        self.index = self._get_index(*self.resized_data.shape)

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        if not self.index.is_trained:
            self.index.train(self.resized_data)

        self.index.add(self.resized_data)

    def denoise(self, queries, noise_var):
        assert self.index is not None
        # queries -= torch.mean(queries, dim=1, keepdim=True)
        queries = torchvision.transforms.Resize((4, 4))(queries.reshape(-1, 1, 8, 8)).reshape(-1, 4 * 4)
        queries_np = np.ascontiguousarray(queries.cpu().numpy(), dtype='float32')
        _, I = self.index.search(queries_np, 1)  # actual search

        NNs = I[:, 0]
        return self.data[NNs].to(queries.device)

def efficient_compute_distances(X, Y):
    """
    Pytorch efficient way of computing distances between all vectors in X and Y, i.e (X[:, None] - Y[None, :])**2
    Get the nearest neighbor index from Y for each X
    :param X:  (n1, d) tensor
    :param Y:  (n2, d) tensor
    Returns a n2 n1 of indices
    """
    dist = (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * torch.mm(X, torch.transpose(Y, 0, 1))
    d = X.shape[1]
    dist /= d # normalize by size of vector to make dists independent of the size of d ( use same alpha for all patche-sizes)
    return dist


def get_NN_indices_low_memory(X, Y, b):
    """
    Get the nearest neighbor index from Y for each X.
    Avoids holding a (n1 * n2) amtrix in order to reducing memory footprint to (b * max(n1,n2)).
    :param X:  (n1, d) tensor
    :param Y:  (n2, d) tensor
    Returns a n2 n1 of indices
    """

    NNs = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
    n_batches = len(X) // b
    for i in range(n_batches):
        dists = efficient_compute_distances(X[i * b:(i + 1) * b], Y)
        NNs[i * b:(i + 1) * b] = dists.min(1)[1]
    if len(X) % b != 0:
        dists = efficient_compute_distances(X[n_batches * b:], Y)
        NNs[n_batches * b:] = dists.min(1)[1]
    return NNs


class NN_Prior:
    def __init__(self, data):
        self.p = int(np.sqrt(data.shape[-1]))
        self.data = data
        self.resized_data = torchvision.transforms.Resize((self.p//2, self.p//2))(data.reshape(-1, 1, self.p, self.p)).reshape(-1, (self.p//2)**2).float()

    def denoise(self, queries, noise_var):
        queries = torchvision.transforms.Resize((self.p//2, self.p//2))(queries.reshape(-1, 1, self.p, self.p)).reshape(-1, (self.p//2)**2).float()

        NNs = get_NN_indices_low_memory(queries, self.resized_data.to(queries.device), b=128)
        return self.data[NNs].to(queries.device)
