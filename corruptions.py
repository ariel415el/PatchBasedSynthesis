import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms


def gauss_kernel(sz: int, sigx: float, sigy: float, corr: float = 0):
    if sz % 2 == 0:
        xx, yy = np.meshgrid(np.arange(-sz // 2, sz // 2), np.arange(-sz // 2, sz // 2))
    else:
        xx, yy = np.meshgrid(np.arange(-sz // 2 + 1, sz // 2 + 1), np.arange(-sz // 2 + 1, sz // 2 + 1))
    xx, yy = xx / sigx, yy / sigy
    kern = np.exp(-.5 * (xx ** 2 + 2 * corr * xx * yy + yy ** 2))
    kern = kern / np.sum(kern, axis=(0, 1))
    return kern


def get_blur_operator(size, sigma):
    kernel = gauss_kernel(size, sigma, sigma)
    # kernel = np.repeat(kernel[None, None], 3, axis=1)
    kernel = kernel[None, None]
    kernel = torch.from_numpy(kernel)

    def convolve(x):
        batched_channels = x.unsqueeze(1)
        k = kernel.to(device=x.device, dtype=x.dtype)
        return F.conv2d(batched_channels, k, padding=size // 2).squeeze(1)

    return convolve


class blur_operator:
    def __init__(self, size, sigma):
        kernel = gauss_kernel(size, sigma, sigma)[None, None]
        self.size = size
        self.kernel = torch.from_numpy(kernel)

    def __call__(self, x):
        batched_channels = x.unsqueeze(1)
        k = self.kernel.to(device=x.device, dtype=x.dtype)
        return F.conv2d(batched_channels, k, padding=self.size // 2).squeeze(1)

    def naive_reverse(self, x):
        return x


class downsample_operator:
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor
        self.down_interpolation = torchvision.transforms.InterpolationMode.BILINEAR
        self.up_interpolation = torchvision.transforms.InterpolationMode.NEAREST

    def __call__(self, x):
        h, w = x.shape[-2:]
        return torchvision.transforms.Resize((int(h * self.scale_factor), int(w * self.scale_factor)), interpolation=self.down_interpolation)(x)

    def naive_reverse(self, x):
        h, w = x.shape[-2:]

        return torchvision.transforms.Resize((int(h / self.scale_factor), int(w / self.scale_factor)), interpolation=self.up_interpolation)(x)

