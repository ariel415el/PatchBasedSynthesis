import os
import joblib
from tqdm import tqdm
import torch
from torchvision.utils import save_image

import sys
sys.path.append("models")
from models.NN_denoiser import MemoryEfficientLocalPatchDenoiser
from utils.image import resize_img, get_patches
from utils.debug_utils import show_images, find_crop_nns

torch.set_grad_enabled(False)


def NNEM(initial_guess, patch_denoiser, patch_size, stride, n_iters=6, n_g_steps=150, lr=1e-2):
    """Peroform EM to minimize the patch NN loss between initial_image and  the implicit prior in the denoiser"""
    patch_weights = 1 # torch.from_numpy(gauss_kernel(patch_size, patch_size, patch_size)).to(initial_guess.device).reshape(1,patch_size**2)
    x = initial_guess.clone().requires_grad_()
    opt = torch.optim.Adam([x], lr=lr)
    pbar = tqdm(range(n_iters))
    for i in pbar:
        z = get_patches(x, patch_size, stride=stride)
        z = patch_denoiser.denoise(z, None)

        with torch.enable_grad():
            for j in range(n_g_steps):
                loss = torch.sum(((get_patches(x, patch_size, stride=stride) - z.detach()) ** 2) * patch_weights)
                opt.zero_grad()
                loss.backward()
                opt.step()
            pbar.set_description(f"Iter: {i}, Step: {j}, Loss: {loss.item():.2f}")

    return x


def MS_NEMM(raw_data,
        initial_image,
        patch_size=8,
        stride=1,
        grayscale=False,
        n_iters=20,
        n_g_steps=3,
        resolutions=[16, 32, 64, 128],
        denoiser_class=MemoryEfficientLocalPatchDenoiser,
        keys_mode=None
):
    """Run multiscale NEMM"""

    debug_images = [(initial_image, "init")]
    lvl_output = initial_image
    for res in resolutions:
        lvl_intitial_guess = resize_img(lvl_output, res)
        denoiser = denoiser_class(raw_data, patch_size, stride,
                                  resize=res, grayscale=grayscale, keys_mode=keys_mode)

        lvl_output = NNEM(lvl_intitial_guess, denoiser, patch_size, stride, n_iters=n_iters, n_g_steps=n_g_steps)

        debug_images.append((lvl_output, f"res - {res}"))

    return lvl_output, debug_images



