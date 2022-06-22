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


def read_local_data(image_paths, patch_size, stride, grayscale=False, img_dim=None):
    """Read 'n_images' random images"""
    data = []
    for i, path in enumerate(image_paths):
        im = load_image(path, grayscale, resize=img_dim)
        h,w = im.shape[-2:]
        data += [get_patches(im, patch_size, stride)]
    data = torch.stack(data, dim=0) # [b, N, c*p*p]
    return data, h, w


def read_random_patches(image_paths, patch_size, stride, samples_per_image=None, grayscale=False, img_dim=None):
    """Draw 'samples_per_image' random patches from 'n_images' random images"""
    data = []
    for i, path in enumerate(image_paths):
        im = load_image(path, grayscale, resize=img_dim)
        patches = get_patches(im, patch_size, stride)
        if samples_per_image is not None:
            idx = random.sample(range(len(patches)), min(len(patches), samples_per_image))
            patches = patches[idx]
        data += [patches]

    data = torch.cat(data, dim=0)
    # data -= torch.mean(data, dim=1, keepdim=True)
    return data


def train_GMM(image_paths, n_components, patch_size, samples_per_image, grayscale=False, img_dim=None):
    data = read_random_patches(image_paths, patch_size, 1, samples_per_image, grayscale=grayscale, img_dim=img_dim)

    # fit a GMM model with EM
    print(f"[*] Fitting GMM with {n_components} to {len(data)} data points..")
    gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='full', verbose=2, verbose_interval=1)
    gmm.fit(data)

    denoiser = GMMDenoiser(pi=torch.from_numpy(gmm.weights_),
                            mu=torch.from_numpy(gmm.means_),
                            sigma=torch.from_numpy(gmm.covariances_), device=torch.device('cpu'))

    denoiser.save(f"saved_models/{denoiser.name}p={patch_size}_c={1 if grayscale else 3}_R={img_dim}_N={len(image_paths)}x{samples_per_image}_.joblib")


def extract_patches(image_paths, patch_size, stride, grayscale=False, img_dim=None):
    patches, h, w = read_local_data(image_paths, patch_size, stride, grayscale=grayscale, img_dim=img_dim)
    patches = patches.permute(1, 0, 2)  #   [N, b, 3*p*p]
    N = len(image_paths)

    # data = {"patches": patches,
    #         "patch_size": patch_size,
    #         "stride": stride,
    #         "channels": 1 if grayscale else 3,
    #         "img_dim": img_dim,
    #         "N": N
    #         }
    joblib.dump(patches, f"saved_models/Patches_p={patch_size}"
                         f"_s={stride}"
                         f"_c={1 if grayscale else 3}"
                         f"_R={img_dim}"
                         f"_N={N}.joblib", protocol=4)


if __name__ == '__main__':
    import json

    data_path = '/cs/labs/yweiss/ariel1/data/FFHQ_128'
    n_images = 1000
    p = 16
    s = 1
    # path_list = os.listdir(data_path)[:n_images]
    path_list = [os.path.join(data_path, name) for name in json.load(open("../top_frontal_facing_FFHQ.txt", 'r'))[:n_images]]

    for img_dim in [16, 32, 64, 128]:
        # for img_dim in [128]:
        # train_GMM(path_list, patch_size=8, samples_per_image=None, n_components=10, grayscale=True, resize=resize)
        extract_patches(path_list, patch_size=p, stride=s, grayscale=True, img_dim=img_dim)
