from tqdm import tqdm

import torch
import sys

from corruptions import downsample_operator
from utils.image import get_patches
sys.path.append("models")
from models.NN_denoiser import LocalPatchDenoiser, NN_Denoiser, MemoryEfficientLocalPatchDenoiser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def optimize_image(x, loss_func, its, lr=1e-2):
    opt = torch.optim.Adam([x], lr=lr)

    x = x.requires_grad_()
    with torch.enable_grad():
        for i in range(its):
            opt.zero_grad()
            loss = loss_func(x)
            loss.backward()
            opt.step()
    return x.data


def decorrupt_with_patch_prior_and_callable_H(corrupt_image, initial_guess, corroption_operator, noise_std, patch_denoiser, betas, patch_size, stride):
    noise_std = noise_std / (patch_size / stride)
    x = initial_guess.clone()
    pbar = tqdm(betas)
    for beta in pbar:
        pbar.set_description(f'beta={beta}')
        z = get_patches(x, patch_size, stride=stride)

        z = patch_denoiser.denoise(z, 1 / beta)

        def aggregation_loss(X):
            global_loss = torch.sum((corroption_operator(X) - corrupt_image) ** 2)
            patch_loss = torch.sum((z - get_patches(X, patch_size, stride=stride)) ** 2)
            return global_loss / noise_std ** 2 + patch_loss * beta

        x = optimize_image(x, aggregation_loss, its=150)

    return x


def MS_EPLL(initial_image, raw_data, resolutions, patch_size, stride, betas, noise_std, grayscale=False, keys_mode=None):
    assert initial_image.shape[-1] == initial_image.shape[-2]

    debug_images = [(initial_image, "init")]
    lvl_output = initial_image.clone()
    for res in resolutions:
        H_full = downsample_operator(initial_image.shape[-1] / res)
        H = downsample_operator(lvl_output.shape[-1] / res)

        denoiser = MemoryEfficientLocalPatchDenoiser(raw_data, patch_size, stride,
                                      resize=res, grayscale=grayscale, keys_mode=keys_mode)

        lvl_intitial_guess = H.naive_reverse(lvl_output)
        lvl_output = decorrupt_with_patch_prior_and_callable_H(initial_image, lvl_intitial_guess, H_full,
                                                            noise_std, denoiser, betas, patch_size, stride)
        debug_images.append((lvl_output, f"res - {res}"))

    return lvl_output, debug_images



