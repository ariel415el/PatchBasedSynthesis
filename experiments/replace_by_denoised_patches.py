import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from corruptions import downsample_operator, blur_operator
from experiments.compare_with_pca import combine_patches
from main import iterative_corruption
from models.GMM_denoiser import GMMDenoiser
from models.NN_denoiser import LocalPatchDenoiser, NN_Denoiser
from utils import load_image, get_patches, show_images

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def combine_patches(patches, patch_size, stride, resize):
    combined = F.fold(patches.T.unsqueeze(0), output_size=resize, kernel_size=patch_size, stride=stride)

    # normal fold matrix
    c = 1
    input_ones = torch.ones((1,c,resize, resize), dtype=patches.dtype, device=patches.device)
    divisor = F.unfold(input_ones, kernel_size=patch_size, dilation=(1, 1), stride=stride, padding=(0, 0))
    divisor = F.fold(divisor, output_size=resize, kernel_size=patch_size, stride=stride)

    divisor[divisor == 0] = 1.0
    return (combined / divisor).squeeze(dim=0)

def replace_by_denoised_patches():
    p = 8
    s = 8
    resize = 128
    grayscale = True
    noise_std = 1 / 255
    local_nn_denoiser = LocalPatchDenoiser.load_from_file(f"../models/saved_models/local_NN_priorp={p}_s={s}_w={16}_c={1 if grayscale else 3}_R={resize}_N=5000.joblib",
                                                          keys_mode=None)
    local_nn_denoiser_resize = LocalPatchDenoiser.load_from_file(f"../models/saved_models/local_NN_priorp={p}_s={s}_w={16}_c={1 if grayscale else 3}_R={resize}_N=5000.joblib",
                                                          keys_mode='resize')
    global_nn_denoiser = NN_Denoiser.load_from_file(f"../models/saved_models/NN_prior_p=8_c={1 if grayscale else 3}_R={resize}_N=100xNone.joblib")
    gmm_denoiser = GMMDenoiser.load_from_file(f"../models/saved_models/GMM(R=128_k=10_(p=8_N=100xNone{'_1C' if grayscale else ''}).joblib",device=device, MAP=True)

    import os
    import json
    image = load_image(os.path.join('../../data/FFHQ_128/', json.load(open("../top_frontal_facing_FFHQ.txt", 'r'))[123]), grayscale=grayscale, resize=resize).to(device)
    # image = load_image('../../data/FFHQ_128/69989.png', grayscale=grayscale, resize=resize).to(device)

    # H = blur_operator(15, 2)
    H = downsample_operator(0.5)

    corrupt_image = iterative_corruption(image, H, noise_std, n_corruptions=1)
    initial_guess = H.naive_reverse(corrupt_image.clone())

    patches = get_patches(initial_guess.clone(), p, s)

    gmm_denoised = gmm_denoiser.denoise(patches, noise_std)
    global_nn_denoised = global_nn_denoiser.denoise(patches, 0)
    local_nn_denoised = local_nn_denoiser.denoise(gmm_denoised, 0)
    local_nn_denoised_resize = local_nn_denoiser_resize.denoise(gmm_denoised, 0)

    gmm_denoised_img = combine_patches(gmm_denoised, p, s, resize)
    global_nn_denoised_img = combine_patches(global_nn_denoised, p, s, resize)
    local_nn_denoised_img = combine_patches(local_nn_denoised, p, s, resize)
    local_nn_denoised_resize_img = combine_patches(local_nn_denoised_resize, p, s, resize)


    for denoised_patches in [
        gmm_denoised,
        global_nn_denoised,
        local_nn_denoised,
        local_nn_denoised_resize
    ]:
        dists = ((denoised_patches - patches)**2)
        print(f"Mean dist : {dists.mean()}, N_exact: {torch.all(dists == 0, dim=1).sum()}")


    # save_image(gmm_denoised.reshape(-1,3,p,p), "gmm_denoised.png", normalize=True)
    show_images([
        (image, 'image'),
        # (corrupt_image, 'corrupt_image'),
        # (initial_guess, 'initial_guess'),
        # (gmm_denoised_img, 'gmm_denoised'),
        # (global_nn_denoised_img, 'global_nn_denoised_img'),
        # (local_nn_denoised_img, 'local_nn_denoised_img'),
        (local_nn_denoised_resize_img, 'local_nn_denoised_resize_img')
    ])

if __name__ == '__main__':
    replace_by_denoised_patches()