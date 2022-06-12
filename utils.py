import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F


def load_image(im_path):
    im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB) / 255.
    return torch.from_numpy(im).permute(2,0,1)


def show_images(images_and_names):
    n = len(images_and_names)
    plt.figure(dpi=200)
    for i, (image, name) in enumerate(images_and_names):
        plt.subplot(1, n, i + 1)
        plt.imshow(np.clip(image.permute(1,2,0).cpu().numpy(), 0, 1))
        plt.title(name)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def get_patches(x, p, stride):
    flat_x = F.unfold(x.unsqueeze(0), kernel_size=p, stride=stride)[0]  # [c*p*p, N]
    return flat_x.T


def split_patches_to_windows(patches, img_height, img_width, patch_size, stride, split_factor=2):
    """Splits a list of patches to a matrix [split_factor x split_factor] of lists of patches. The split is by spatial location"""
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