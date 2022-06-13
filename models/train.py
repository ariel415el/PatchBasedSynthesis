import os
import random

import joblib
import numpy as np
import torch
from sklearn import mixture

from NN_denoiser import FaissNNModule, windowed_prior_wrapper, NN_Prior
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
    out_path = f"saved_models/GMM(k={n_components}_(p={patch_size}_N={n_images}x{samples_per_image}).joblib"
    joblib.dump(gmm, out_path)


def train_global_nn_denoiser():
    n_images = 1000
    samples_per_image = 1000
    patch_size = 16
    stride = 2
    data = read_random_patches(data_path, patch_size, stride, n_images, samples_per_image)

    print(f"[*] Creating NN prior for {len(data)} data points..")
    # nn_prior = FaissNNModule()
    # nn_prior.set_index(data)
    nn_prior = NN_Prior(data)
    joblib.dump(nn_prior, f"saved_models/nn_global_(p={patch_size}_N={n_images}x{samples_per_image}).joblib")


def train_local_nn_denoisers():
    n_images = 1000
    patch_size = 8
    stride = 3
    n_windows_per_dim = 8
    patches, h, w = read_local_data(data_path, n_images, patch_size, stride)
    patches = patches.permute(1,0,2)

    patch_indices = patch_to_window_index(patch_size, stride, h, w, n_windows_per_dim)

    nn_priors = []
    for i in sorted(np.unique(patch_indices)):
        data = patches[patch_indices == i].reshape(-1, patches.shape[-1])
        # nn_priors.append(FaissNNModule())
        # nn_priors[-1].set_index(data)
        nn_priors.append(NN_Prior(data))

    prior = windowed_prior_wrapper(nn_priors, patch_indices)

    joblib.dump(prior, f"saved_models/nn_local_(N={n_images}_p={patch_size}_s={stride}_w={n_windows_per_dim}).joblib")


if __name__ == '__main__':
    data_path = '/mnt/storage_ssd/datasets/FFHQ_128'

    train_GMM()
    # train_local_nn_denoisers()
    # train_global_nn_denoiser()
