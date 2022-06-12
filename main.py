import joblib
import torch

from EPLL import decorrupt_with_patch_prior_and_callable_H
from corruptions import downsample_operator, blur_operator
from utils import load_image, show_images
from models.GMM_denoiser import GMMDenoiser

noise_std = 1 / 255
alpha = 1 / 50
betas = [min(2 ** i / alpha, 3000) for i in range(6)]
patch_size = 8
stride = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


denoiser = GMMDenoiser('models/saved_models/GMM(k=10)_SkLearn(p=8_N=1000x1000).joblib', device, MAP=True)

# denoiser = joblib.load('nn_prior(p=8_N=1000x1000).joblib')

image = load_image('/mnt/storage_ssd/datasets/FFHQ_128/69999.png').to(device) #.float()

# H = blur_operator(15, 2)
H = downsample_operator(0.5)

n_corruptions = 1
corrupt_image = image
for i in range(n_corruptions):
    corrupt_image = H(corrupt_image)
corrupt_image += noise_std * torch.randn_like(corrupt_image, device=device)

MAP = corrupt_image
for i in range(n_corruptions):
    MAP = decorrupt_with_patch_prior_and_callable_H(MAP, noise_std=noise_std, H=H, patch_denoiser=denoiser, betas=betas, patch_size=patch_size, stride=stride)

show_images([(image, 'image'), (corrupt_image, 'corrupt'), (MAP, 'MAP')])

