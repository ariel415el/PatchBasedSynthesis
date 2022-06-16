import os
import random

import joblib
import numpy as np
import torch
from sklearn import mixture
import torchvision.transforms.functional as F
from NN_denoiser import LocalPatchDenoiser, NN_Prior
from models.GMM_denoiser import GMMDenoiser
from utils import get_patches, load_image, patch_to_window_index


def read_local_data(data_path, patch_size, stride, n_images, grayscale=False, resize=None):
    """Read 'n_images' random images"""
    # img_list = random.sample(os.listdir(data_path), n_images)
    img_list = os.listdir(data_path)[:n_images]
    data = []
    for i, im_name in enumerate(img_list):
        im = load_image(os.path.join(data_path, im_name), grayscale, resize)
        h,w = im.shape[-2:]
        data += [get_patches(im, patch_size, stride)]
    data = torch.stack(data, dim=0) # [b, N, c*p*p]
    return data, h, w


def read_random_patches(data_path, patch_size, stride, n_images, samples_per_image=None, grayscale=False, resize=None):
    """Draw 'samples_per_image' random patches from 'n_images' random images"""
    # img_list = random.sample(os.listdir(data_path), n_images)
    img_list = os.listdir(data_path)[:n_images]
    data = []
    for i, im_name in enumerate(img_list):
        im = load_image(os.path.join(data_path, im_name), grayscale, resize)
        patches = get_patches(im, patch_size, stride)
        if samples_per_image is not None:
            idx = random.sample(range(len(patches)), min(len(patches), samples_per_image))
            patches = patches[idx]
        data += [patches]

    data = torch.cat(data, dim=0)
    # data -= torch.mean(data, dim=1, keepdim=True)
    return data


def train_GMM(data_path, n_components, patch_size, n_images, samples_per_image, grayscale=False, resize=None):
    data = read_random_patches(data_path, patch_size, 1, n_images, samples_per_image, grayscale=grayscale, resize=resize)

    # fit a GMM model with EM
    print(f"[*] Fitting GMM with {n_components} to {len(data)} data points..")
    gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='full', verbose=2, verbose_interval=1)
    gmm.fit(data)

    denoiser = GMMDenoiser(pi=torch.from_numpy(gmm.weights_),
                            mu=torch.from_numpy(gmm.means_),
                            sigma=torch.from_numpy(gmm.covariances_), device=torch.device('cpu'))

    denoiser.save(f"saved_models/GMM(R={resize}_k={n_components}_(p={patch_size}_N={n_images}x{samples_per_image}{'_1C' if grayscale else ''}).joblib")


def train_global_nn_denoiser(data_path, patch_size, n_images, samples_per_image, grayscale=False, resize=None):
    data = read_random_patches(data_path, patch_size, 1, n_images, samples_per_image, grayscale=grayscale, resize=resize)
    print(f"[*] Creating NN prior for {len(data)} data points..")
    nn_prior = NN_Prior(data, patch_size, 1 if grayscale else 3)
    nn_prior.save(f"saved_models/nn_global(R={resize}_p={patch_size}_N={n_images}x{samples_per_image}{'_1C' if grayscale else ''}).joblib")


def train_local_nn_denoisers(data_path, patch_size, stride, n_images, n_windows_per_dim=None, grayscale=False, resize=None):

    patches, h, w = read_local_data(data_path, patch_size, stride, n_images, grayscale=grayscale, resize=resize)
    patches = patches.permute(1, 0, 2)  #   [N, b, 3*p*p]

    patch_index_to_widnow_index = None
    data_sets = []
    if n_windows_per_dim:
        patch_index_to_widnow_index = patch_to_window_index(patch_size, stride, h, w, n_windows_per_dim)
        for i in sorted(np.unique(patch_index_to_widnow_index)):
            data = patches[patch_index_to_widnow_index == i].reshape(-1, patches.shape[-1])
            data_sets.append(data)
    else:
        for i in range(len(patches)):
            data_sets.append(patches[i])

    prior = LocalPatchDenoiser(data_sets, patch_size, 1 if grayscale else 3, patch_index_to_widnow_index)

    prior.save(f"saved_models/nn_local(R={resize}_N={n_images}_p={patch_size}_s={stride}_w={n_windows_per_dim}{'_1C' if grayscale else ''}).joblib")


if __name__ == '__main__':
    data_path = '/mnt/storage_ssd/datasets/FFHQ_128'

    for resize in [16, 32, 64, 128]:
        train_GMM(data_path, n_images=100, patch_size=8, samples_per_image=None, n_components=10, grayscale=True, resize=resize)
        # train_local_nn_denoisers(data_path, patch_size=8, stride=1, n_images=1000, n_windows_per_dim=2, grayscale=True, resize=resize)
        # train_global_nn_denoiser(data_path, patch_size=8, n_images=500, samples_per_image=None, grayscale=True, resize=resize)
