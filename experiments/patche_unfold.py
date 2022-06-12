import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from utils import load_image, get_patches

def main():
    p = 32
    s = 16
    image = load_image('/mnt/storage_ssd/datasets/FFHQ_128/69999.png')
    h,w = image.shape[-2:]

    patches = get_patches(image, p, s).reshape(-1,3,p,p)

    n_windows_per_dim = 2
    patch_sets = split_patches_to_windows(patches, h, w, p, s, 2)
    for r in range(n_windows_per_dim):
        for c in range(n_windows_per_dim):
            window_patches = patch_sets[r][c]
            save_image(window_patches, f"set_{r}x{c}.png", normalize=True, nrow=int(np.ceil(np.sqrt(len(window_patches)))))

    save_image(patches, "asd.png", normalize=True)


def split_patches_to_windows(patches, img_height, img_width, patch_size, stride, split_factor=2):
    patch_sets = []
    for r in range(split_factor):
        row = []
        for c in range(split_factor):
            row.append([])
        patch_sets.append(row)

    n_patches_in_row = (img_height - patch_size) // stride + 1
    n_patches_in_col = (img_width - patch_size) // stride + 1
    for i, patch in enumerate(patches):
        patch_row = i // n_patches_in_row
        patch_col = i % n_patches_in_col

        window_row = min(split_factor - 1, patch_row // (n_patches_in_row // split_factor))
        window_col = min(split_factor - 1, patch_col // (n_patches_in_col // split_factor))
        patch_sets[window_row][window_col].append(patch)

    return patch_sets


if __name__ == '__main__':
    main()