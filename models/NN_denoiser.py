import joblib
import numpy as np
import torch
import torchvision.transforms as tv_t
import torch.nn.functional as F
from utils import get_patches


class LocalPatchDenoiser:
    def __init__(self, raw_data, patch_size, stride, window_size, grayscale=True, resize=None, keys_mode=None):
        print("Local patch Denoiser:")
        if grayscale:
            raw_data = torch.mean(raw_data, dim=1, keepdim=True)
        if resize is not None:
            raw_data = tv_t.Resize((resize, resize))(raw_data)
            print("\t(*) data resized")
        if keys_mode == 'context':
            hp = patch_size // 4
            raw_data = F.pad(raw_data, (hp, hp, hp, hp))
            print("\t(*) data resized")
            self.window_indices = LocalPatchDenoiser._set_patch_window_indices(resize, patch_size//2, stride, window_size)
        else:
            self.window_indices = LocalPatchDenoiser._set_patch_window_indices(resize, patch_size, stride, window_size)

        self.data = F.unfold(raw_data, kernel_size=patch_size, stride=stride).permute(2, 0, 1)  # (#img, d, #patch) -> (#patch, #img, d)

        self.priors = []
        for i in np.unique(self.window_indices):
            local_data = self.data[self.window_indices == i].reshape(-1, self.data.shape[-1])
            self.priors.append(NN_Denoiser(local_data, patch_size, raw_data.shape[1], keys_mode))
        print("\t(*) Local priors loaded")

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

    def denoise(self, queries, noise_var):
        queries_copy = queries.clone()
        for i in range(len(self.priors)):
            queries_copy[self.window_indices == i] = self.priors[i].denoise(queries_copy[self.window_indices == i], None)
        return queries_copy



class MemoryEfficientLocalPatchDenoiser:
    def __init__(self, raw_data, patch_size, stride, grayscale=True, resize=None, keys_mode=None):
        print("Local patch Denoiser:")
        if grayscale:
            raw_data = torch.mean(raw_data, dim=1, keepdim=True)
        if resize is not None:
            raw_data = tv_t.Resize((resize, resize))(raw_data)
            print("\t(*) data resized")

        self.data = raw_data

        self.n_patches_in_dim = (resize - patch_size) // stride + 1
        self.patch_size = patch_size

    def get_local_data(self, patch_index):
        row = patch_index // self.n_patches_in_dim
        col = patch_index % self.n_patches_in_dim

        return self.data[..., row: row + self.patch_size, col: col + self.patch_size].reshape(len(self.data), -1)

    def denoise(self, queries, noise_var):
        queries_copy = queries.clone()
        for i in range(len(queries)):
            local_data = self.get_local_data(i).short()
            NNs = get_NN_indices_low_memory(queries_copy[i].unsqueeze(0).short(), local_data, b=512)
            queries_copy[i] = local_data[NNs]
        return queries_copy

class NN_Denoiser:
    def __init__(self, data, patch_size, channels, keys_mode=None):
        self.c = channels
        self.p = patch_size
        self.data = data
        self.keys_mode = keys_mode
        if keys_mode == 'resize':
            self.resize = tv_t.Resize((self.p//2, self.p//2), tv_t.InterpolationMode.NEAREST)
            self.resized_data = self.resize(data.reshape(-1, self.c, self.p, self.p)).reshape(-1, self.c*(self.p//2)**2).float()

        elif keys_mode == 'PCA':
            self.data_mean = data.mean(0)
            U, S, V = torch.svd((data - self.data_mean).T @ (data - self.data_mean))
            k = int(np.sqrt(data.shape[1]))
            self.projection_matrix = V[:, :k]
            # import matplotlib.pyplot as plt; plt.plot(np.arange(len(S)), S); plt.show()
            self.projected_data = (data - self.data_mean) @ self.projection_matrix
        self.name = f"NN_prior"

    def denoise(self, queries, noise_var):
        b = 256
        if self.keys_mode == 'resize':
            resize_queries = self.resize(queries.reshape(-1, self.c, self.p, self.p)).float().reshape(-1, self.c*(self.p//2)**2)
            NNs = get_NN_indices_low_memory(resize_queries, self.resized_data.to(resize_queries.device), b=b)
        elif self.keys_mode == 'PCA':
            NNs = get_NN_indices_low_memory((queries - self.data_mean) @ self.projection_matrix.to(queries.device), self.projected_data.to(queries.device), b=b)
        else:
            NNs = get_NN_indices_low_memory(queries, self.data.to(queries.device), b=b)

        return self.data[NNs].to(queries.device)


def efficient_compute_distances(X, Y, r=0.8):
    """
    Pytorch efficient way of computing distances between all vectors in X and Y, i.e (X[:, None] - Y[None, :])**2
    Get the nearest neighbor index from Y for each X
    :param X:  (n1, d) tensor
    :param Y:  (n2, d) tensor
    Returns a n2 n1 of indices
    """
    if r == 2:
        dist = (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * torch.mm(X, torch.transpose(Y, 0, 1))
    else:
        dist = torch.norm(X[:, None] - Y[None,], p=r, dim=-1)
    dist /= X.shape[1]  # normalize by size of vector to make dists independent of the size of d ( use same alpha for all patche-sizes)
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

