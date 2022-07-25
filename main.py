import joblib
import torch

from models.GMM_denoiser import GMMDenoiser
from models.NN_denoiser import MemoryEfficientLocalPatchDenoiser
from EPLL import MS_EPLL
from NNEM import MS_NEMM
from utils.debug_utils import find_crop_nns, show_images
from utils.image import resize_img

def epll_synthesis():
    noise_std = 1 / 255
    alpha = 1 / 50
    betas = [min(2 ** i / alpha, 3000) for i in range(6)]

    return MS_EPLL(input_image, raw_data, resolutions, patch_size, stride, betas, noise_std, grayscale,  keys_mode)

def nnem_synthesis():
    n_iters = 10
    return MS_NEMM(raw_data,
                input_image,
                patch_size=patch_size,
                n_iters=n_iters,
                resolutions=resolutions,
                denoiser_class=MemoryEfficientLocalPatchDenoiser,
                keys_mode=keys_mode
                )

if __name__ == '__main__':
    method = "NNEM"  # EPLL / NNEM
    start_from = "GMM_sample"  # GMM_sample / mean
    limit_data = 1000
    dtype = torch.float32
    device = torch.device("cuda:0")
    min_dim = 8
    resolutions = [16, 24, 32, 48, 64]
    show_NNs = True
    keys_mode = 'PCA'
    patch_size = 8
    stride = 1
    grayscale = False

    raw_data = joblib.load(f"models/saved_models/Frontal_FFHQ_N=5000.joblib")
    raw_data = raw_data[torch.randperm(raw_data.size(0))[:limit_data]].to(dtype).to(device)

    if start_from == "mean":
        input_image = torch.mean(raw_data[torch.randperm(raw_data.size(0))[:100]], dim=0, keepdim=False)
        input_image = resize_img(input_image, min_dim).to(device)
    else:
        sampler = joblib.load(f"models/saved_models/GMM-k=100_r={min_dim}.joblib")
        input_image = sampler.sample()[0][0].reshape(3, min_dim, min_dim)
        input_image = torch.from_numpy(input_image).to(device).to(dtype)
        if grayscale:
            input_image = torch.mean(input_image, dim=0, keepdim=True)

    if method == "EPLL":
        output, debug_images = epll_synthesis()
    else:
        output, debug_images = nnem_synthesis()

    if show_NNs:
        debug_images += find_crop_nns(output, raw_data)

    show_images(debug_images, f"outputs/{method}_from_{start_from}.png")

