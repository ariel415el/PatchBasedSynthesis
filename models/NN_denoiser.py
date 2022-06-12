import faiss
import numpy as np
import torch


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
        resized_data = torchvision.transforms.Resize((4,4))(data.reshape(-1,1, 8,8)).reshape(-1,4*4)
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
        import torchvision
        queries = torchvision.transforms.Resize((4, 4))(queries.reshape(-1, 1, 8, 8)).reshape(-1, 4 * 4)
        queries_np = np.ascontiguousarray(queries.cpu().numpy(), dtype='float32')
        _, I = self.index.search(queries_np, 1)  # actual search

        NNs = I[:, 0]
        return self.data[NNs].to(queries.device)

class NN_Prior(Denoiser):
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        self.index = None

    def _get_index(self, n, d):
        return faiss.IndexIVFFlat(faiss.IndexFlat(d), d, int(np.sqrt(n)))
        # return faiss.IndexIVFPQ(faiss.IndexFlatL2(d), d, int(np.sqrt(n)), 8, 8)

    def set_index(self, data):
        self.index_vectors = np.ascontiguousarray(data.numpy(), dtype='float32')
        self.index = self._get_index(*self.index_vectors.shape)

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        if not self.index.is_trained:
            self.index.train(self.index_vectors)

        self.index.add(self.index_vectors)

    def denoise(self, queries, noise_var):
        assert self.index is not None
        queries -= torch.mean(queries, dim=1, keepdim=True)

        queries_np = np.ascontiguousarray(queries.cpu().numpy(), dtype='float32')
        _, I = self.index.search(queries_np, 1)  # actual search

        NNs = I[:, 0]
        return torch.from_numpy(self.index_vectors[NNs]).to(queries.device)