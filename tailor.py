# import json
# import os

import joblib
import torch
import torchvision.transforms as tv_t
from tqdm import tqdm

from NN_tailoring import tailor_image
from corruptions import downsample_operator
from utils import load_image, show_images
import sys
sys.path.append("models")
from models.NN_denoiser import LocalPatchDenoiser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_grad_enabled(False)


def iterative_corruption(image, H, noise_std, n_corruptions):
    corrupt_image = image
    for i in range(n_corruptions):
        corrupt_image = H(corrupt_image)
    corrupt_image += noise_std * torch.randn_like(corrupt_image, device=corrupt_image.device)
    return corrupt_image


def main():
    noise_std = 0 # 1 / 255
    patch_size = 8
    stride = 1
    img_dim = 128
    n_levels = 10
    pyr_factor = 0.75
    window_dim = 1
    grayscale = True
    H = downsample_operator(pyr_factor)

    raw_data = joblib.load(f"models/saved_models/Frontal_FFHQ_N=5000.joblib")

    image_path = '../data//FFHQ_128/69989.png'
    image = load_image(image_path, grayscale=grayscale, resize=img_dim).to(device)

    resolutions = []
    corrupt_image = image.clone()
    for i in range(n_levels):
        resolutions = [corrupt_image.shape[-1]] + resolutions
        corrupt_image = H(corrupt_image)
    corrupt_image += noise_std * torch.randn_like(corrupt_image, device=corrupt_image.device)

    debug_images = [
        (image, 'input'),
        (corrupt_image, 'corrupt_image'),
        # (H.naive_reverse(corrupt_image), 'initial-guess'),
    ]
    lvl_output = corrupt_image
    for res in resolutions:
        lvl_intitial_guess = H.naive_reverse(lvl_output, res)
        denoiser = LocalPatchDenoiser(raw_data, patch_size, stride,
                                      resize=res, grayscale=grayscale, window_size=window_dim, keys_mode='resize')

        lvl_output = tailor_image(lvl_intitial_guess, denoiser, patch_size, stride, n_iters=6, n_g_steps=150)

        debug_images.append((lvl_output, f"res - {res}"))

    # debug_images.append((tv_t.Resize((resize, resize))(corrupt_image), 'bilinear-upsample'))

    nn1 = find_global_nn(lvl_output, tv_t.Resize((img_dim, img_dim))(raw_data))
    debug_images.append((nn1, 'train NN to output'))

    show_images(debug_images)
    return lvl_output


def find_global_nn(img1, data):

    if img1.shape[0] == 1:
        img1 = img1.repeat(3,1,1)

    print("finding VGG NN")
    from torchvision import models, transforms
    vgg_features = models.vgg19(pretrained=True).features
    vgg_features.eval()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])

    layers = [4, 9, 18]#, 27, 36]
    features_1 = [vgg_features[:l](normalize(img1).float()) for l in layers]

    def find_nns(query_features_list):
        n = len(query_features_list)
        min_dist = [torch.inf]*n
        min_idx = [-1]*n
        for i in tqdm(range(len(data))):
            for j, l in enumerate(layers):
                ref_features = vgg_features[:l](normalize(data[i]).float())
                for q in range(n):
                    dist = torch.norm(ref_features - query_features_list[q][j])
                    if dist < min_dist[q]:
                        min_dist[q] = dist
                        min_idx[q] = i
        return min_idx

    nns = find_nns([features_1])
    return data[nns[0]]

if __name__ == '__main__':
    main()
