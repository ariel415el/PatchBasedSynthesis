import torch
from tqdm import tqdm

from utils import get_patches


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

