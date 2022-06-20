# import faiss
import joblib
import numpy as np
import torch
import torchvision
import torchvision.transforms as tv_t

class LocalPatchDenoiser:
    def __init__(self, data_sets, patch_size, channels, patch_index_to_widnow_index=None, keys_mode=None):
        print("Loading local priors...")
        self.priors = [NN_Denoiser(data, patch_size, channels, keys_mode=keys_mode) for data in data_sets]
        print("Done...")
        self.p = patch_size
        self.c = channels
        if patch_index_to_widnow_index is None:
            patch_index_to_widnow_index = list(range(len(data_sets)))
        self.patch_indices = patch_index_to_widnow_index
        self.name = f"local_NN_prior"

    def denoise(self, queries, noise_var):
        queries_copy = queries.clone()
        for i in range(len(self.priors)):
            queries_copy[self.patch_indices == i] = self.priors[i].denoise(queries_copy[self.patch_indices == i], None)
        return queries_copy

    def save(self, path):
        d = {
            'patch_size': self.p,
            'channels': self.c,
            'data_sets': [p.data for p in self.priors],
            'patch_index_to_widnow_index': self.patch_indices
        }
        joblib.dump(d, path, protocol=4)

    @staticmethod
    def load_from_file(path, keys_mode=None):
        d = joblib.load(path)
        return LocalPatchDenoiser(**d, keys_mode=keys_mode)
#
# class FaissNNModule:
#     def __init__(self, use_gpu=False):
#         self.use_gpu = use_gpu
#         self.index = None
#
#     def _get_index(self, n, d):
#         # return faiss.IndexIVFFlat(faiss.IndexFlat(d), d, int(np.sqrt(n)))
#         return faiss.IndexIVFPQ(faiss.IndexFlatL2(d), d, int(np.sqrt(n)), 8, 8)
#
#     def set_index(self, data):
#         import torchvision
#         self.data = data
#         self.c = 3
#         self.p = int(np.sqrt(data.shape[-1] // self.c))
#         self.resized_data = tv_t.Resize((self.p//2, self.p//2))(data.reshape(-1, self.c, self.p, self.p)).reshape(-1, self.c*(self.p//2)**2).float()
#         self.resized_data = np.ascontiguousarray(self.resized_data.numpy(), dtype='float32')
#         self.index = self._get_index(*self.resized_data.shape)
#
#         if self.use_gpu:
#             res = faiss.StandardGpuResources()
#             self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
#
#         if not self.index.is_trained:
#             self.index.train(self.resized_data)
#
#         self.index.add(self.resized_data)
#
#     def denoise(self, queries, noise_var):
#         assert self.index is not None
#         # queries -= torch.mean(queries, dim=1, keepdim=True)
#
#         resize_queries = tv_t.Resize((self.p//2, self.p//2))(queries.reshape(-1, 3, self.p, self.p)).reshape(-1, self.c*(self.p//2)**2).float()
#         resize_queries = np.ascontiguousarray(resize_queries.cpu().numpy(), dtype='float32')
#
#         _, I = self.index.search(resize_queries, 1)  # actual search
#
#         NNs = I[:, 0]
#         return self.data[NNs].to(queries.device)
#
#     def save(self, path):
#         joblib.dump(self.data, path)
#
#     @staticmethod
#     def load_from_file(path, use_gpu=False):
#         data = joblib.load(path)
#         prior = FaissNNModule(use_gpu)
#         prior.set_index(data)
#         return prior


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

def get_NN_indices_low_memory_2(X, Y, X2, Y2, b):
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
        dists += efficient_compute_distances(X2[i * b:(i + 1) * b], Y2)
        NNs[i * b:(i + 1) * b] = dists.min(1)[1]
    if len(X) % b != 0:
        dists = efficient_compute_distances(X[n_batches * b:], Y)
        dists += efficient_compute_distances(X2[n_batches * b:], Y2)
        NNs[n_batches * b:] = dists.min(1)[1]
    return NNs


class NN_Denoiser:
    def __init__(self, data, patch_size, channels, keys_mode=None):
        self.c = channels
        self.p = patch_size
        self.data = data
        self.keys_mode = keys_mode
        self.resize = tv_t.Resize((self.p//2, self.p//2), tv_t.InterpolationMode.NEAREST)
        if keys_mode == 'resize':
            self.resized_data = self.resize(data.reshape(-1, self.c, self.p, self.p)).reshape(-1, self.c*(self.p//2)**2).float()

        elif keys_mode == 'PCA':
            self.data_mean = data.mean(0)
            U, S, V = torch.svd((data - self.data_mean).T @ (data - self.data_mean))
            k = data.shape[1] // 8
            self.projection_matrix = V[:, :k]
            self.projected_data = (data - self.data_mean) @ self.projection_matrix
            print("Pca done")
        else:
            print("NN_Denoiser: NN data Load")
        self.name = f"NN_prior"

    def denoise(self, queries, noise_var):
        b = 256
        if self.keys_mode == 'resize':
            resize_queries = self.resize(queries.reshape(-1, self.c, self.p, self.p)).float().reshape(-1, self.c*(self.p//2)**2)
            NNs = get_NN_indices_low_memory(resize_queries, self.resized_data.to(resize_queries.device),
                                              b=b)
        elif self.keys_mode == 'PCA':
            NNs = get_NN_indices_low_memory((queries - self.data_mean) @ self.projection_matrix.to(queries.device), self.projected_data.to(queries.device), b=b)
        else:
            NNs = get_NN_indices_low_memory(queries, self.data.to(queries.device), b=b)
        return self.data[NNs].to(queries.device)

    def save(self, path):
        joblib.dump({'data': self.data, 'patch_size': self.p, 'channels': self.c}, path)

    @staticmethod
    def load_from_file(path, keys_mode=None):
        d = joblib.load(path)
        return NN_Denoiser(**d, keys_mode=keys_mode)