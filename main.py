import json
import os

import joblib
import torch
import torchvision.transforms as tv_t
from matplotlib import pyplot as plt

from EPLL import decorrupt_with_patch_prior_and_callable_H
from corruptions import downsample_operator, blur_operator, RepeteadOperator
from utils import load_image, show_images, plot_img
import sys
sys.path.append("models")
from models.GMM_denoiser import GMMDenoiser
from models.NN_denoiser import LocalPatchDenoiser, NN_Denoiser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


def iterative_corruption(image, H, noise_std, n_corruptions):
    corrupt_image = image
    for i in range(n_corruptions):
        corrupt_image = H(corrupt_image)
    corrupt_image += noise_std * torch.randn_like(corrupt_image, device=corrupt_image.device)
    return corrupt_image

def main():
    noise_std = 1 / 255
    alpha = 1 / 50
    betas = [min(2 ** i / alpha, 3000) for i in range(6)]
    patch_size = 8
    stride = 1
    n_levels = 3
    img_dim = 64
    window_dim = 1
    grayscale = True
    # H = blur_operator(15, 2)
    H = downsample_operator(0.5)

    raw_data = joblib.load(f"models/saved_models/Frontal_FFHQ_N=5000.joblib")

    # image_path = '../data//FFHQ_128/69989.png'
    image_path = f'../data/FFHQ_128/{json.load(open("top_frontal_facing_FFHQ.txt", "r"))[123]}'
    image = load_image(image_path, grayscale=grayscale, resize=img_dim).to(device)

    corrupt_image = iterative_corruption(image, H, noise_std, n_corruptions=n_levels)

    outputs = []

    tmp_img = corrupt_image
    for i in range(n_levels):
        resolution = img_dim//2**(n_levels - 1 - i)
        # denoiser = GMMDenoiser.load_from_file(f"models/saved_models/GMM(R={resolution}_k=10_(p=8_N=100xNone{'_1C' if grayscale else ''}).joblib", device=device, MAP=True)
        # denoiser = NN_Denoiser.load_from_file(f"models/saved_models/NN_prior_p=8_c={1 if grayscale else 3}_R={resolution}_N=100xNone.joblib")

        denoiser = LocalPatchDenoiser(raw_data, patch_size, stride,
                                      resize=resolution, grayscale=grayscale, window_size=window_dim, keys_mode='PCA')

        lvl_intitial_guess = H.naive_reverse(tmp_img)

        # Compare to smallest image
        lvl_curropt_image = corrupt_image
        lvl_corroption_operator = RepeteadOperator(H, i + 1)

        # Compare to last level output
        # lvl_curropt_image = tmp_img
        # lvl_corroption_operator = H

        tmp_img = decorrupt_with_patch_prior_and_callable_H(lvl_curropt_image, lvl_intitial_guess, lvl_corroption_operator,
                                                            noise_std, denoiser, betas, patch_size, stride)
        outputs.append(tmp_img)

    debug_images = [
        (image, 'input'),
        (corrupt_image, 'corrupt_image'),
        # (H.naive_reverse(corrupt_image), 'initial-guess'),
    ]

    for i in range(len(outputs)):
        res = img_dim//2**(len(outputs) - 1 - i)
        debug_images.append((outputs[i], f'output - {res}'))

    # debug_images.append((tv_t.Resize((resize, resize))(corrupt_image), 'bilinear-upsample'))
    show_images(debug_images)

if __name__ == '__main__':
    main()