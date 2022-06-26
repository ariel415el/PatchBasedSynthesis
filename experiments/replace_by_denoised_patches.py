import joblib
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import os
import json
from corruptions import downsample_operator, blur_operator
from experiments.compare_with_pca import combine_patches
from main import iterative_corruption
from models.GMM_denoiser import GMMDenoiser
from models.NN_denoiser import LocalPatchDenoiser, NN_Denoiser
from utils import load_image, get_patches, show_images

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def combine_patches(patches, patch_size, stride, img_dim):
    combined = F.fold(patches.T.unsqueeze(0), output_size=img_dim, kernel_size=patch_size, stride=stride)
    # normal fold matrix
    c = 1
    input_ones = torch.ones((1,c,img_dim, img_dim), dtype=patches.dtype, device=patches.device)
    divisor = F.unfold(input_ones, kernel_size=patch_size, dilation=(1, 1), stride=stride, padding=(0, 0))
    divisor = F.fold(divisor, output_size=img_dim, kernel_size=patch_size, stride=stride)

    divisor[divisor == 0] = 1.0
    return (combined / divisor).squeeze(dim=0)

def combine_patches_median(patches, patch_size, stride, img_dim):
    resized_patches = patches.reshape(-1, patch_size, patch_size)
    pixel_arryas = [[] for x in range(img_dim**2)]
    n_patches_with_stride = (img_dim - patch_size) // stride + 1
    for patch_index in range(n_patches_with_stride ** 2):
        for i in range(patch_size):
            for j in range(patch_size):
                row = (patch_index // n_patches_with_stride)*stride + i
                col = (patch_index % n_patches_with_stride)*stride + j
                idx = row * img_dim + col
                pixel_arryas[idx].append(resized_patches[patch_index, i, j].item())

    resized_patches = np.array([np.median(x) for x in pixel_arryas]).reshape(1, img_dim, img_dim)
    return torch.from_numpy(resized_patches)

def replace_by_denoised_patches():
    p = 16
    s = 16
    resize = 64
    grayscale = True
    noise_std = 1 / 255
    raw_data = joblib.load(f"../models/saved_models/Frontal_FFHQ_N=5000.joblib")

    denoisers = [
        # (NN_Denoiser(f"../models/saved_models/Patches_p={p}_s=1_c=1_R={resize}_N=1000.joblib",
        #              p, 1, keys_mode=None),
        #  'global_nn'),
        # (LocalPatchDenoiser(f"../models/saved_models/Patches_p={p}_s=1_c=1_R={resize}_N=1000.joblib",
        #                     p, s, 1, window_size=1, img_dim=resize, keys_mode=None),
        #  'local_nn'),
        # (LocalPatchDenoiser(f"../models/saved_models/Patches_p={p}_s=1_c=1_R={resize}_N=1000.joblib",
        #                     p, s, 1, window_size=1, img_dim=resize, keys_mode='resize'),
        #  'local_nn_resize'),
        (LocalPatchDenoiser(raw_data, p, s, resize=resize, grayscale=grayscale, window_size=1, keys_mode='PCA'),
        'local_nn_PCA'),
    ]

    image = load_image(os.path.join('../../data/FFHQ_128/', json.load(open("../top_frontal_facing_FFHQ.txt", 'r'))[123]), grayscale=grayscale, resize=resize).to(device)
    # image = load_image('../../data/FFHQ_128/69989.png', grayscale=grayscale, resize=resize).to(device)
    H = downsample_operator(0.5)

    corrupt_image = iterative_corruption(image, H, noise_std, n_corruptions=1)
    initial_guess = H.naive_reverse(corrupt_image.clone())
    corrupt_patches = get_patches(initial_guess.clone(), p, s)

    debug_pairs = [(image, 'image'), (corrupt_image, 'corrupt image')]
    original_patches = get_patches(image, p, s)
    for denoiser, name in denoisers:
        denoised_patches = denoiser.denoise(corrupt_patches, 0)
        #
        dists = ((denoised_patches - original_patches)**2)
        tmp = (combine_patches(denoised_patches, p, s, resize),
               f"{name}: avg-dist:{dists.mean():.4f}, exact:{(dists.sum(1) == 0).sum()} / {len(dists)}")

        # denoised_patches = denoised_patches.reshape(-1,p,p)[:, p//4:3*p//4, p//4:3*p//4].reshape(-1,p**2//4)
        # tmp = (combine_patches(denoised_patches, p//2, s, resize-p//2), "asd")

        # padded_input = F.pad(initial_guess, (p//4, p//4, p//4, p//4))
        # corrupt_patches = get_patches(padded_input.clone(), p, s)
        # denoised_patches = denoiser.denoise(corrupt_patches, 0).reshape(-1,p,p)[:, p//4:3*p//4, p//4:3*p//4].reshape(-1,p**2//4)
        # tmp = (combine_patches(denoised_patches, p//2, s, resize), "asd")

        debug_pairs.append(tmp)

    show_images(debug_pairs)


if __name__ == '__main__':
    replace_by_denoised_patches()