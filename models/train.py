import os
import random

import joblib
import numpy as np
import torch
from sklearn import mixture

from NN_denoiser import LocalPatchDenoiser, NN_Prior
from models.GMM_denoiser import GMMDenoiser
from utils import get_patches, load_image, patch_to_window_index


def read_local_data(data_path, n_images, patch_size, stride):
    img_list = random.sample(os.listdir(data_path), n_images)
    data = []
    for i, im_name in enumerate(img_list):
        im = load_image(os.path.join(data_path, im_name))
        gray_im = torch.mean(im, dim=0, keepdim=True)
        h,w = gray_im.shape[-2:]
        data += [get_patches(gray_im, patch_size, stride)]
    data = torch.stack(data, dim=0) # [b, N, 3*p*p]
    return data, h, w


def read_random_patches(data_path, patch_size, stride, n_images, samples_per_image=None, substruct_mean=False):
    img_list = random.sample(os.listdir(data_path), n_images)
    data = []
    for i, im_name in enumerate(img_list):
        im = load_image(os.path.join(data_path, im_name))
        gray_im = torch.mean(im, dim=0, keepdim=True)
        patches = get_patches(gray_im, patch_size, stride)
        if samples_per_image is not None:
            patches = patches[np.random.randint(len(patches), size=samples_per_image)]
        data += [patches]

    data = torch.cat(data, dim=0).float()
    if substruct_mean:
        data -= torch.mean(data, dim=1, keepdim=True)
    return data


def train_GMM():
    n_images = 1000
    samples_per_image = 1000
    patch_size = 16
    stride = 2
    n_components = 10

    data = read_random_patches(data_path, patch_size, stride, n_images, samples_per_image)

    # fit a GMM model with EM
    print(f"[*] Fitting GMM with {n_components} to {len(data)} data points..")
    gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='full', verbose=2, verbose_interval=1)
    gmm.fit(data)

    denoiser = GMMDenoiser(pi=torch.from_numpy(gmm.weights_),
                            mu=torch.from_numpy(gmm.means_),
                            sigma=torch.from_numpy(gmm.covariances_))

    denoiser.save(f"saved_models/GMM(k={n_components}_(p={patch_size}_N={n_images}x{samples_per_image}).joblib")


def train_global_nn_denoiser(data_path, n_images, patch_size, stride, samples_per_image):
    data = read_random_patches(data_path, patch_size, stride, n_images, samples_per_image)
    print(f"[*] Creating NN prior for {len(data)} data points..")
    nn_prior = NN_Prior(data)
    nn_prior.save(f"saved_models/nn_global_(p={patch_size}_N={n_images}x{samples_per_image}).joblib")


def train_local_nn_denoisers(data_path, n_images, patch_size, stride, n_windows_per_dim=None):

    patches, h, w = read_local_data(data_path, n_images, patch_size, stride)
    patches = patches.permute(1, 0, 2)  #   [N, b, 3*p*p]

    patch_index_to_widnow_index = None
    nn_priors = []
    if n_windows_per_dim:
        patch_index_to_widnow_index = patch_to_window_index(patch_size, stride, h, w, n_windows_per_dim)
        for i in sorted(np.unique(patch_index_to_widnow_index)):
            data = patches[patch_index_to_widnow_index == i].reshape(-1, patches.shape[-1])
            nn_priors.append(NN_Prior(data))
    else:
        for i in range(len(patches)):
            nn_priors.append(patches[i])

    prior = LocalPatchDenoiser(nn_priors, patch_index_to_widnow_index)

    prior.save("saved_models/nn_local_(N={n_images}_p={patch_size}_s={stride}_w={n_windows_per_dim}).joblib")


if __name__ == '__main__':
    data_path = '/cs/labs/yweiss/ariel1/data/FFHQ_128'

    # train_GMM()
    train_local_nn_denoisers(data_path, n_images=100, patch_size=8, stride=2, n_windows_per_dim=None)
    train_global_nn_denoiser(data_path, n_images=100, patch_size=8, stride=2, samples_per_image=None)
