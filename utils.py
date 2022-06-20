import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F
import torchvision.transforms as tv_t

def load_image(im_path, grayscale=False, resize=None):
    im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB) / 255.
    im = torch.from_numpy(im).permute(2,0,1)
    if resize is not None:
        im = tv_t.Resize((resize, resize))(im)
    if grayscale:
        im = torch.mean(im, dim=0, keepdim=True)
    return im


def plot_img(axs, img, name):
    axs.imshow(np.clip(img.permute(1,2,0).cpu().numpy(), 0, 1), cmap='gray')
    axs.set_title(name, size=5)
    axs.axis('off')


def show_images(images_and_names):
    fig, axs = plt.subplots(1, len(images_and_names), dpi=200, figsize=(2*len(images_and_names),2))
    for i, (image, name) in enumerate(images_and_names):
        plot_img(axs[i], image, name)
    plt.tight_layout()
    plt.show()


def get_patches(x, p, stride):
    flat_x = F.unfold(x.unsqueeze(0), kernel_size=p, stride=stride)[0]  # [c*p*p, N]
    return flat_x.T # .reshape(flat_x.shape[1], -1, p, p)

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

