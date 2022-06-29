import numpy as np
import torch
import torchvision.transforms as tv_t
import torch.nn.functional as F


class LocalPatchDenoiser:
    def __init__(self, raw_data, patch_size, stride, grayscale=True, resize=None, keys_mode=None):
        print("Local patch Denoiser:")
        if grayscale:
            raw_data = torch.mean(raw_data, dim=1, keepdim=True)
        if resize is not None:
            raw_data = tv_t.Resize((resize, resize))(raw_data)
            print("\t(*) data resized")

        data = F.unfold(raw_data, kernel_size=patch_size, stride=stride).permute(2, 0, 1)  # (#img, d, #patch) -> (#patch, #img, d)

        self.n_patches_in_dim = (resize - patch_size) // stride + 1
        self.patch_size = patch_size

        self.priors = []
        for i in range(self.n_patches_in_dim**2):
            local_data = data[i].reshape(-1, data.shape[-1])
            self.priors.append(NN_Denoiser(local_data, patch_size, raw_data.shape[1], keys_mode))
        print("\t(*) Local priors loaded")

    def denoise(self, queries, noise_var):
        queries_copy = queries.clone()
        for i in range(len(self.priors)):
            queries_copy[i] = self.priors[i].denoise(queries_copy[i].unsqueeze(0), None)
        return queries_copy


class MemoryEfficientLocalPatchDenoiser:
    def __init__(self, raw_data, patch_size, stride, grayscale=True, resize=None, keys_mode=None):
        print("Local patch Denoiser:")
        if grayscale:
            raw_data = torch.mean(raw_data, dim=1, keepdim=True)
        if resize is not None:
            raw_data = tv_t.Resize((resize, resize))(raw_data)
            print("\t(*) data resized")

        self.keys_mode = keys_mode

        self.data = raw_data.contiguous()

        self.n_patches_in_dim = (resize - patch_size) // stride + 1
        self.patch_size = patch_size

    def __str__(self):
        return "MemoryEfficientLocalPatchDenoiser"

    def denoise(self, queries, noise_var):
        queries_copy = queries.clone()
        for i in range(len(queries)):
            row = i // self.n_patches_in_dim
            col = i % self.n_patches_in_dim
            queries_copy_i = queries_copy[i].unsqueeze(0)
            refs = self.data[..., row: row + self.patch_size, col: col + self.patch_size].reshape(len(self.data), -1)
            if self.keys_mode == "rand":
                rand = torch.randn(refs.shape[1], int(np.sqrt(refs.shape[-1]))).to(refs.device)  # (slice_size**2*ch)
                queries_copy_i = queries_copy_i @ rand
                refs = refs @ rand
                # rand = rand / torch.std(rand, dim=0, keepdim=True)  # noramlize

            NN = efficient_compute_distances(queries_copy_i, refs, r=2).min(1)[1].item()
            queries_copy[i] = self.data[..., row: row + self.patch_size, col: col + self.patch_size][NN].reshape(-1)
        return queries_copy


class ParallelLocalPatchDenoiser:
    def __init__(self, raw_data, patch_size, stride, grayscale=True, resize=None, keys_mode=None):
        print("Local patch Denoiser:")
        if grayscale:
            raw_data = torch.mean(raw_data, dim=1, keepdim=True)
        if resize is not None:
            raw_data = tv_t.Resize((resize, resize))(raw_data)
            print("\t(*) data resized")

        self.data = F.unfold(raw_data, kernel_size=patch_size, stride=stride).permute(2, 0, 1)  # (#img, d, #patch) -> (#patch, #img, d)

        self.n_patches_in_dim = (resize - patch_size) // stride + 1
        self.patch_size = patch_size
        self.keys_mode = keys_mode
        if keys_mode == 'PCA':
            self.projection_matrices = []
            self.means = []
            self.projected_data = []
            self.d = int(np.sqrt(self.data.shape[-1]))
            for i in range(len(self.data)):
                projection_matrix, data_mean = learn_pca(self.data[i], d=self.d)
                self.projection_matrices.append(projection_matrix)
                self.means.append(data_mean)
                self.projected_data.append((self.data[i] - data_mean) @ projection_matrix)

            self.projection_matrices = torch.stack(self.projection_matrices)
            self.projected_data = torch.stack(self.projected_data)
            self.means = torch.stack(self.means)

            self.Y_sq = (self.projected_data * self.projected_data).sum(-1)[:, None]
            self.Y = self.projected_data.permute(0, 2, 1)
        else:
            self.Y_sq = (self.data * self.data).sum(-1)[:, None]
            self.Y = self.data.permute(0, 2, 1)

    def __str__(self):
        return "ParallelLocalPatchDenoiser"

    def denoise(self, queries, noise_var):
        if self.keys_mode == 'PCA':
            X = ((queries - self.means).unsqueeze(-1).repeat(1, 1, self.d) * self.projection_matrices.to(queries.device)).sum(1).unsqueeze(1)
        else:
            X = queries.unsqueeze(1)
        NNs = ((X * X).sum(-1)[:, None] + self.Y_sq - 2.0 * X @ self.Y).min(-1)[1].squeeze(1)
        return self.data[np.arange(len(self.data)), NNs]

def learn_pca(data, d):
    data_mean = data.mean(0)
    U, S, V = torch.svd((data - data_mean).T @ (data - data_mean))
    projection_matrix = V[:, :d]
    return projection_matrix, data_mean

class NN_Denoiser:
    def __init__(self, data, patch_size, channels, keys_mode=None):
        self.c = channels
        self.p = patch_size
        self.data = data
        self.keys_mode = keys_mode
        if keys_mode == 'resize':
            self.resize = tv_t.Resize((self.p//2, self.p//2), tv_t.InterpolationMode.NEAREST)
            self.resized_data = self.resize(data.reshape(-1, self.c, self.p, self.p)).reshape(-1, self.c*(self.p//2)**2).to(self.data.dtype)
        elif keys_mode == 'PCA':
            self.projection_matrix, self.data_mean = learn_pca(data, d=int(np.sqrt(data.shape[1])))
            self.projected_data = (data - self.data_mean) @ self.projection_matrix
        self.name = f"NN_prior"

    def denoise(self, queries, noise_var):
        b = 256
        if self.keys_mode == 'resize':
            resize_queries = self.resize(queries.reshape(-1, self.c, self.p, self.p)).reshape(-1, self.c*(self.p//2)**2).to(self.data.dtype)
            NNs = get_NN_indices_low_memory(resize_queries, self.resized_data.to(resize_queries.device), b=b)
        elif self.keys_mode == 'PCA':
            NNs = get_NN_indices_low_memory((queries - self.data_mean) @ self.projection_matrix.to(queries.device), self.projected_data.to(queries.device), b=b)
        else:
            NNs = get_NN_indices_low_memory(queries, self.data.to(queries.device), b=b)

        return self.data[NNs].to(queries.device)


def efficient_compute_distances(X, Y, r=2):
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

