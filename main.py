import joblib
import numpy as np
import torch
from matplotlib import pyplot as plt

from EPLL import decorrupt_with_patch_prior_and_callable_H
from corruptions import downsample_operator, blur_operator
from utils import load_image, show_images, plot_multi_scale_decorruption
import sys
sys.path.append("models")
from models.GMM_denoiser import GMMDenoiser
from models.NN_denoiser import LocalPatchDenoiser, NN_Prior, FaissNNModule

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
    n_levels = 1
    resize = 128
    grayscale = True

    H = blur_operator(15, 2)
    # H = downsample_operator(0.5)

    image = load_image('/mnt/storage_ssd/datasets/FFHQ_128/69993.png', grayscale=grayscale, resize=resize).to(device)

    corrupt_image = iterative_corruption(image, H, noise_std, n_corruptions=n_levels)

    outputs = []
    initial_guesses = []

    tmp_img = corrupt_image
    for i in range(n_levels):
        resolution = resize//2**(n_levels - 1 - i)
        denoiser = GMMDenoiser.load_from_file(f"models/saved_models/GMM(R={resolution}_k=10_(p=8_N=100xNone{'_1C' if grayscale else ''}).joblib", device=device, MAP=True)
        # denoiser = LocalPatchDenoiser.load_from_file("models/saved_models/nn_local(N=1000_p=8_s=2_w=None).joblib")
        # denoiser = LocalPatchDenoiser.load_from_file(f"models/saved_models/nn_local(R={resolution}_N=1000_p=8_s=1_w=2{'_1C' if grayscale else ''}).joblib")

        tmp_img, debug = decorrupt_with_patch_prior_and_callable_H(tmp_img, noise_std, H, denoiser, betas, patch_size, stride)
        outputs.append(tmp_img)
        initial_guesses.append(debug)

    plot_multi_scale_decorruption(outputs, image, corrupt_image, initial_guesses)

if __name__ == '__main__':
    main()