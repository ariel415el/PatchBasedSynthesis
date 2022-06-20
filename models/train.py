import os
import random
import sys

import joblib
import numpy as np
import torch
from sklearn import mixture
import torchvision.transforms.functional as F
sys.path.append(os.path.dirname(__file__))
from NN_denoiser import LocalPatchDenoiser, NN_Denoiser
from models.GMM_denoiser import GMMDenoiser
from utils import get_patches, load_image, patch_to_window_index


def read_local_data(image_paths, patch_size, stride, grayscale=False, resize=None):
    """Read 'n_images' random images"""
    data = []
    for i, path in enumerate(image_paths):
        im = load_image(path, grayscale, resize)
        h,w = im.shape[-2:]
        data += [get_patches(im, patch_size, stride)]
    data = torch.stack(data, dim=0) # [b, N, c*p*p]
    return data, h, w


def read_random_patches(image_paths, patch_size, stride, samples_per_image=None, grayscale=False, resize=None):
    """Draw 'samples_per_image' random patches from 'n_images' random images"""
    data = []
    for i, path in enumerate(image_paths):
        im = load_image(path, grayscale, resize)
        patches = get_patches(im, patch_size, stride)
        if samples_per_image is not None:
            idx = random.sample(range(len(patches)), min(len(patches), samples_per_image))
            patches = patches[idx]
        data += [patches]

    data = torch.cat(data, dim=0)
    # data -= torch.mean(data, dim=1, keepdim=True)
    return data


def train_GMM(image_paths, n_components, patch_size, samples_per_image, grayscale=False, resize=None):
    data = read_random_patches(image_paths, patch_size, 1, samples_per_image, grayscale=grayscale, resize=resize)

    # fit a GMM model with EM
    print(f"[*] Fitting GMM with {n_components} to {len(data)} data points..")
    gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='full', verbose=2, verbose_interval=1)
    gmm.fit(data)

    denoiser = GMMDenoiser(pi=torch.from_numpy(gmm.weights_),
                            mu=torch.from_numpy(gmm.means_),
                            sigma=torch.from_numpy(gmm.covariances_), device=torch.device('cpu'))

    denoiser.save(f"saved_models/{denoiser.name}p={patch_size}_c={1 if grayscale else 3}_R={resize}_N={len(image_paths)}x{samples_per_image}_.joblib")


def train_global_nn_denoiser(image_paths, patch_size, samples_per_image, grayscale=False, resize=None):
    data = read_random_patches(image_paths, patch_size, 1, samples_per_image, grayscale=grayscale, resize=resize)
    print(f"[*] Creating NN prior for {len(data)} data points..")
    denoiser = NN_Denoiser(data, patch_size, 1 if grayscale else 3)

    denoiser.save(f"saved_models/{denoiser.name}_p={patch_size}_c={1 if grayscale else 3}_R={resize}_N={len(image_paths)}x{samples_per_image}.joblib")


def train_local_nn_denoisers(image_paths, patch_size, stride, n_windows_per_dim=None, grayscale=False, resize=None):

    patches, h, w = read_local_data(image_paths, patch_size, stride, grayscale=grayscale, resize=resize)
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

    denoiser = LocalPatchDenoiser(data_sets, patch_size, 1 if grayscale else 3, patch_index_to_widnow_index)

    denoiser.save(f"saved_models/{denoiser.name}p={patch_size}_s={stride}_w={n_windows_per_dim}_c={1 if grayscale else 3}_R={resize}_N={len(image_paths)}.joblib")


if __name__ == '__main__':
    import json

    data_path = '/cs/labs/yweiss/ariel1/data/FFHQ_128'
    n_images = 5000
    p = 8
    s = 8
    # path_list = os.listdir(data_path)[:n_images]
    path_list = [os.path.join(data_path, name) for name in json.load(open("../top_frontal_facing_FFHQ.txt", 'r'))[:n_images]]

    # for resize in [16, 32, 64, 128]:
    for resize in [128]:
        # train_GMM(path_list, patch_size=8, samples_per_image=None, n_components=10, grayscale=True, resize=resize)
        train_local_nn_denoisers(path_list, patch_size=p, stride=s, n_windows_per_dim=(resize//s)//2, grayscale=True, resize=resize)
        # train_global_nn_denoiser(path_list, patch_size=8, samples_per_image=None, grayscale=True, resize=resize)
