import os
import random

import joblib
import numpy as np
import torch
from sklearn import mixture

from models.denoiser import FaissNNModule
from utils import get_patches, load_image


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
    patch_size = 8
    stride = 2
    n_components = 3

    data = read_random_patches(data_path, patch_size, stride, n_images, samples_per_image)

    # fit a GMM model with EM
    print(f"[*] Fitting GMM with {n_components} to {len(data)} data points..")
    gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='full', verbose=2, verbose_interval=1)
    gmm.fit(data)
    out_path = f"saved_models/GMM(k={n_components}_(p={patch_size}_N={n_images}x{samples_per_image}).joblib"
    joblib.dump(gmm, out_path)


def train_approximate_nn(data, descriptor):
    print(f"[*] Creating NN prior for {len(data)} data points..")
    nn_prior = FaissNNModule()
    nn_prior.set_index(data)
    joblib.dump(nn_prior, f"nn_prior({descriptor}).joblib")


if __name__ == '__main__':
    data_path = '/mnt/storage_ssd/datasets/FFHQ_128'

    train_GMM()
    # train_approximate_nn(data, descriptor)
