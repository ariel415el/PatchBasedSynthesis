# import faiss
import joblib
import numpy as np
import torch
import torchvision
import torchvision.transforms as tv_t


class LocalPatchDenoiser:
    def __init__(self, data_path, patch_size, stride, channels, window_size, img_dim=None, keys_mode=None):
        print("Loading local priors...")
        raw_data = joblib.load(data_path)

        if stride > 1:
            strided_indices = LocalPatchDenoiser._get_strided_data(img_dim, patch_size, stride)
            raw_data = raw_data[strided_indices]

        print("Done...")

        self.window_indices = LocalPatchDenoiser._set_patch_window_indices(img_dim, patch_size, stride, window_size)

        self.priors = []
        for i in np.unique(self.window_indices):
            local_data = raw_data[self.window_indices == i]
            self.priors.append(NN_Denoiser(local_data, patch_size, channels, keys_mode))

    @staticmethod
    def _set_patch_window_indices(img_dim, patch_size, stride, window_size):
        window_indices = []
        n_patches_in_dim = (img_dim - patch_size) // stride + 1
        for patch_index in  range(n_patches_in_dim ** 2):
            patch_row = patch_index // n_patches_in_dim
            patch_col = patch_index % n_patches_in_dim

            window_row = patch_row // window_size
            window_col = patch_col // window_size
            window_index = window_row * n_patches_in_dim + window_col
            window_indices.append(window_index)

        return np.array(window_indices)

    @staticmethod
    def _get_strided_data(img_dim, patch_size, stride):
        n_patches_with_stride_1 = img_dim - patch_size + 1
        strided_indices_mask = []
        for patch_index in range(n_patches_with_stride_1 ** 2):
            patch_row = patch_index // n_patches_with_stride_1
            patch_col = patch_index % n_patches_with_stride_1
            strided_indices_mask.append((patch_row % stride == 0) and (patch_col % stride == 0))
        return np.array(strided_indices_mask)

    def denoise(self, queries, noise_var):
        queries_copy = queries.clone()
        for i in range(len(self.priors)):
            queries_copy[self.window_indices == i] = self.priors[i].denoise(queries_copy[self.window_indices == i], None)
        return queries_copy


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
        if type(data) == str:
            data = joblib.load(data)
        data = data.reshape(-1, data.shape[-1])
        self.data = data
        self.keys_mode = keys_mode
        self.resize = tv_t.Resize((self.p//2, self.p//2), tv_t.InterpolationMode.NEAREST)
        if keys_mode == 'resize':
            self.resized_data = self.resize(data.reshape(-1, self.c, self.p, self.p)).reshape(-1, self.c*(self.p//2)**2).float()

        elif keys_mode == 'PCA':
            self.data_mean = data.mean(0)
            U, S, V = torch.svd((data - self.data_mean).T @ (data - self.data_mean))
            k = int(np.sqrt(data.shape[1]))
            self.projection_matrix = V[:, :k]
            # import matplotlib.pyplot as plt; plt.plot(np.arange(len(S)), S); plt.show()
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
