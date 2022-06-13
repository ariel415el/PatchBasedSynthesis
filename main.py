import joblib
import torch

from EPLL import decorrupt_with_patch_prior_and_callable_H
from corruptions import downsample_operator, blur_operator
from utils import load_image, show_images
import sys
sys.path.append("models")
from models.GMM_denoiser import GMMDenoiser
from models.NN_denoiser import LocalPatchDenoiser, NN_Prior

noise_std = 1 / 255
alpha = 1 / 50
betas = [min(2 ** i / alpha, 3000) for i in range(6)]
patch_size = 8
stride = 1
n_corruptions = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


# denoiser = GMMDenoiser.load_from_file('models/saved_models/GMM100.mdl', device=device, MAP=True)

# denoiser = LocalPatchDenoiser.load_from_file("models/saved_models/nn_local_(N={n_images}_p={patch_size}_s={stride}_w={n_windows_per_dim}).joblib")
denoiser = NN_Prior.load_from_file("models/saved_models/nn_global_(p=8_N=100xNone).joblib")

image = load_image('/cs/labs/yweiss/ariel1/data/FFHQ_128/69999.png').to(device)

H = blur_operator(15, 2)
# H = downsample_operator(0.5)

corrupt_image = image
for i in range(n_corruptions):
    corrupt_image = H(corrupt_image)
corrupt_image += noise_std * torch.randn_like(corrupt_image, device=device)

MAP = corrupt_image
for i in range(n_corruptions):
    MAP = decorrupt_with_patch_prior_and_callable_H(MAP, noise_std=noise_std, H=H, patch_denoiser=denoiser, betas=betas, patch_size=patch_size, stride=stride)

show_images([(image, 'image'), (corrupt_image, 'corrupt'), (MAP, 'MAP')])

