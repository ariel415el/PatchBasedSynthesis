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


def tailor_image(raw_data,
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

    debug_images = []
    lvl_output = initial_image
    for res in resolutions:
        lvl_intitial_guess = resize_img(lvl_output, res)
        denoiser = denoiser_class(raw_data, patch_size, stride,
                                  resize=res, grayscale=grayscale, keys_mode=keys_mode)

        lvl_output = NNEM(lvl_intitial_guess, denoiser, patch_size, stride, n_iters=n_iters, n_g_steps=n_g_steps)

        # debug_images.append((lvl_output, f"res - {res}"))
    debug_images.append((lvl_output, f"final"))

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


def interpolate():
    raw_data = joblib.load(f"models/saved_models/Frontal_FFHQ_N=5000.joblib")

    # Get initial image
    gmm = joblib.load(f"models/saved_models/GMM-k=100.joblib")
    initial_image_1 = torch.from_numpy(gmm.sample(1)[0][0].reshape(3, 8, 8))
    initial_image_2 = torch.from_numpy(gmm.sample(1)[0][0].reshape(3, 8, 8))

    n_steps = 100
    for i in range(n_steps + 1):
        initial_image = initial_image_1 * (1-(i/float(n_steps))) + initial_image_2 * (i/float(n_steps))

        output, debug_images = tailor_image(raw_data,
                                    initial_image,
                                    patch_size=8,
                                    device=torch.device("cuda:0"),
                                    n_iters=10,
                                    resolutions=[16, 32, 64, 128],
                                    limit_images=1000,
                                    denoiser_class=MemoryEfficientLocalPatchDenoiser,
                                    keys_mode=None
                            )
        out_dir = f"outputs_new/interpolations-{n_steps}"
        os.makedirs(out_dir, exist_ok=True)
        show_images(debug_images, f"{out_dir}/debug-{i}.png")


def start_from_mean(n_images=1):
    for i in range(n_images):
        dtype = torch.float32
        device = torch.device("cuda:0")
        data_size = 1000
        init_res = 8
        patch_size = 8
        raw_data = joblib.load(f"models/saved_models/Frontal_FFHQ_N=5000.joblib")
        data_subset = raw_data[torch.randperm(raw_data.size(0))[:data_size]].to(dtype).to(device)

        initial_image = torch.mean(raw_data[torch.randperm(raw_data.size(0))[:10]], dim=0, keepdim=False)
        initial_image = resize_img(initial_image, init_res).to(dtype).to(device)

        output, debug_images = tailor_image(data_subset,
                                            initial_image,
                                            patch_size=patch_size,
                                            n_iters=10,
                                            resolutions=list(range(max(patch_size*2,init_res), 80, 8)),
                                            denoiser_class=MemoryEfficientLocalPatchDenoiser,
                                            keys_mode=None
                                            )

        debug_images += find_crop_nns(output, data_subset, device)
        debug_images.append((initial_image, "initial image"))

        show_images(debug_images, f"outputs_new/start_from_mean-from-{init_res}-{i}.png")

if __name__ == '__main__':
    start_from_mean(5)