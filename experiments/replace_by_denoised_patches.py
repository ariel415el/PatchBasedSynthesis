import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from corruptions import downsample_operator, blur_operator
from main import iterative_corruption
from models.GMM_denoiser import GMMDenoiser
from models.NN_denoiser import LocalPatchDenoiser, NN_Denoiser
from utils import load_image, get_patches, show_images

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def replace_by_denoised_patches():
    p = 8
    s = 8
    resize = 128
    grayscale = True
    noise_std = 1 / 255
    # global_nn_denoiser = LocalPatchDenoiser.load_from_file(f"../models/saved_models/nn_local(R={resize}_N=1000_p=8_s=8_w=2{'_1C' if grayscale else ''}).joblib")
    global_nn_denoiser = NN_Denoiser.load_from_file(f"../models/saved_models/nn_global(R={resize}_p=8_N=500xNone{'_1C' if grayscale else ''}).joblib")
    gmm_denoiser = GMMDenoiser.load_from_file(f"../models/saved_models/GMM(R=128_k=10_(p=8_N=100xNone{'_1C' if grayscale else ''}).joblib",device=device, MAP=True)

    image = load_image('/mnt/storage_ssd/datasets/FFHQ_128/69992.png', grayscale=grayscale, resize=resize).to(device)

    # H = blur_operator(15, 2)
    H = downsample_operator(0.5)

    corrupt_image = iterative_corruption(image, H, noise_std, n_corruptions=1)
    initial_guess = H.naive_reverse(corrupt_image)
    # initial_guess = corrupt_image.clone()

    patches = get_patches(initial_guess, p, s)

    gmm_denoised = gmm_denoiser.denoise(patches, noise_std)
    nn_denoised = global_nn_denoiser.denoise(patches, 0)
    gmm_plus_nn_denoised = global_nn_denoiser.denoise(gmm_denoised, 0)

    gmm_denoised_img = F.fold(gmm_denoised.reshape(patches.shape[0], -1).T.unsqueeze(0), kernel_size=p, stride=s, output_size=resize)[0]
    nn_denoised_img = F.fold(nn_denoised.reshape(patches.shape[0], -1).T.unsqueeze(0), kernel_size=p, stride=s, output_size=resize)[0]
    gmm_plus_nn_denoised_img = F.fold(gmm_plus_nn_denoised.reshape(patches.shape[0], -1).T.unsqueeze(0), kernel_size=p, stride=s, output_size=resize)[0]


    # save_image(gmm_denoised.reshape(-1,3,p,p), "gmm_denoised.png", normalize=True)
    show_images([
        (image, 'image'),
        # (corrupt_image, 'corrupt_image'),
        (initial_guess, 'corrupt_initial_guess_image'),
        (gmm_denoised_img, 'gmm_denoised'),
        (nn_denoised_img, 'nn_denoised_img'),
        # (gmm_plus_nn_denoised_img, 'gmm_plus_nn_denoised_img')
    ])

if __name__ == '__main__':
    replace_by_denoised_patches()