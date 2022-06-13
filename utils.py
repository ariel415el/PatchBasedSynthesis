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


def patch_to_window_index(patch_size, stride, img_height, img_width, n_windows_per_dim):
    n_patches_in_row = (img_height - patch_size) // stride + 1
    n_patches_in_col = (img_width - patch_size) // stride + 1

    patch_indices = []
    for i in range( n_patches_in_row * n_patches_in_col):
        patch_row = i // n_patches_in_row
        patch_col = i % n_patches_in_col

        window_row = min(n_windows_per_dim - 1, patch_row // (n_patches_in_row // n_windows_per_dim))
        window_col = min(n_windows_per_dim - 1, patch_col // (n_patches_in_col // n_windows_per_dim))

        patch_indices.append(window_row*n_windows_per_dim + window_col)

    return np.array(patch_indices)

