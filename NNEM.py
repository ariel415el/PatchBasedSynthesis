import torch
from tqdm import tqdm

from corruptions import gauss_kernel
from utils import get_patches
import torch.nn.functional as F

def tailor_image(initial_guess, patch_denoiser, patch_size, stride, n_iters=6, n_g_steps=150, lr=1e-2):
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
            pbar.set_description(f"Iter: {i}, Step: {j}, Loss: {loss.item()}")

    return x


# def combine_patches(patches, patch_size, stride, img_dim):
#     combined = F.fold(patches.T.unsqueeze(0), output_size=img_dim, kernel_size=patch_size, stride=stride)
#     # normal fold matrix
#     c = 1
#     input_ones = torch.ones((1,c,img_dim, img_dim), dtype=patches.dtype, device=patches.device)
#     divisor = F.unfold(input_ones, kernel_size=patch_size, dilation=(1, 1), stride=stride, padding=(0, 0))
#     divisor = F.fold(divisor, output_size=img_dim, kernel_size=patch_size, stride=stride)
#
#     divisor[divisor == 0] = 1.0
#     return (combined / divisor).squeeze(dim=0)
#
#
# def tailor_image_patches(initial_guess, patch_denoiser, patch_size, stride, n_iters=150, n_g_steps=10, lr=1e-2):
#     x = get_patches(initial_guess, patch_size, stride=stride).requires_grad_()
#     opt = torch.optim.Adam([x], lr=lr)
#     pbar = tqdm(range(n_iters))
#     for i in pbar:
#         z = patch_denoiser.denoise(x, None)
#
#         with torch.enable_grad():
#             for j in range(n_g_steps):
#                 loss = torch.sum((x - z.detach()) ** 2)
#                 opt.zero_grad()
#                 loss.backward()
#                 opt.step()
#                 pbar.set_description(f"Iter: {i}, Step: {j}, Loss: {loss.item()}")
#
#     return combine_patches(x, patch_size, stride, initial_guess.shape[-1])


