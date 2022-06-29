import os
import joblib
import numpy as np
from tqdm import tqdm
import itertools
import torch
from torchvision.utils import save_image
import torchvision.transforms as tv_t

import sys
sys.path.append("models")
from models.NN_denoiser import MemoryEfficientLocalPatchDenoiser
from utils.image import resize_img, get_patches
from utils.debug_utils import find_global_neural_nn, show_images

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


def tailor_image(patch_size=8,
        stride=1,
        grayscale=False,
        dtype=torch.float32,
        device=torch.device("cuda:0"),
        n_iters=20,
        n_g_steps=3,
        resolutions=[16, 32, 64, 128],
        limit_images=1000,
        denoiser_class=MemoryEfficientLocalPatchDenoiser,
        keys_mode=None
):
    """Run multiscale NEMM"""
    
    # Get train data
    raw_data = joblib.load(f"models/saved_models/Frontal_FFHQ_N=5000.joblib").to(dtype).to(device)
    raw_data = raw_data[torch.randperm(raw_data.size(0))[:limit_images]]

    # raw_data = get_derivative(raw_data, 1)

    # Get initial image
    gmm = joblib.load(f"models/saved_models/GMM-k=100.joblib")
    initial_image = torch.from_numpy(gmm.sample(1)[0][0].reshape(3, 8, 8)).to(dtype).to(device)
    # initial_image = torch.randn((3, 8, 8)).to(dtype).to(device)

    # initial_image = get_derivative(initial_image.unsqueeze(0), 1)[0]

    # debug_images = [(initial_image, 'initial_image')]
    debug_images = []

    lvl_output = initial_image
    for res in resolutions:
        lvl_intitial_guess = resize_img(lvl_output, res)
        denoiser = denoiser_class(raw_data, patch_size, stride,
                                  resize=res, grayscale=grayscale, keys_mode='rand')

        lvl_output = NNEM(lvl_intitial_guess, denoiser, patch_size, stride, n_iters=n_iters, n_g_steps=n_g_steps)

        # debug_images.append((lvl_output, f"res - {res}"))
    debug_images.append((lvl_output, f"final"))

    # Debug: show VGG NN
    img_dim = resolutions[-1]
    raw_data = tv_t.Resize((img_dim, img_dim))(raw_data)
    d = img_dim // 4

    all_slices = np.arange(d,img_dim-d, d)
    all_slices = list(zip(all_slices, all_slices + d))
    all_pairs_of_slices = itertools.product(all_slices, repeat=2)
    for i, (row_interval, col_interval) in enumerate(all_pairs_of_slices,
    ):
        nn_indices = find_global_neural_nn(lvl_output, raw_data, row_interval, col_interval, device)
        nn = raw_data[nn_indices[0]].clone() * 0.5
        nn[:, row_interval[0]:row_interval[1], col_interval[0]: col_interval[1]] *= 2
        debug_images.append((nn, f'crop vgg {(row_interval, col_interval)} -NN'))

    return lvl_output, debug_images


def run_in_batches():
    N_samples = 100
    for patch_size, n_train_images, n_iters, denoiser_class, keys_mode, device in [
        (8, 1000, 10, MemoryEfficientLocalPatchDenoiser, None, torch.device("cuda:0")),
        # (4, 100, 10, MemoryEfficientLocalPatchDenoiser, None, torch.device("cuda:0")),
        # (8, 100, 50, MemoryEfficientLocalPatchDenoiser, None, torch.device("cuda:0")),
        # (8, 100, 10, ParallelLocalPatchDenoiser, "PCA", torch.device("cpu")),
        # (4, 5000, 25, ParallelLocalPatchDenoiser, "PCA", torch.device("cpu")),
    ]:
        out_dir = f"outputs_new/p={patch_size}_N={n_train_images}_iters={n_iters}_K={keys_mode}"
        os.makedirs(out_dir, exist_ok=True)
        for i in range(N_samples):
            output, debug_images = tailor_image(
                patch_size=patch_size,
                device=device,
                n_iters=n_iters,
                resolutions=[16, 32, 64, 128],
                limit_images=n_train_images,
                denoiser_class=denoiser_class,
                keys_mode=keys_mode
            )
            save_image(output, f"{out_dir}/{i}.png", normalize=True)
    # show_images(debug_images)

def single_run(path=None):
    output, debug_images = tailor_image(
        patch_size=8,
        device=torch.device("cuda:0"),
        n_iters=10,
        # resolutions=[16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 128],
        resolutions=[16, 24, 32, 48, 64, 96, 128],
        # resolutions=[16, 32, 64, 128],
        limit_images=1000,
        denoiser_class=MemoryEfficientLocalPatchDenoiser,
        keys_mode=None
    )
    show_images(debug_images, path)

if __name__ == '__main__':
    # single_run()
    for i in range(20):
        single_run(f"outputs/output-{i}.png")
    # run_in_batches()