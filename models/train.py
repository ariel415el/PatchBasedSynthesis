import os
import sys
import torch

import joblib
from sklearn import mixture

sys.path.append(os.path.dirname(__file__))
from utils import load_image
import json
import torchvision.transforms as tv_t



def train_low_res_GMM(data, n_components, grayscale=False, img_dim=8):
    # fit a GMM model with EM
    if grayscale:
        data = torch.mean(data, dim=1, keepdim=True)
    data = tv_t.Resize((img_dim, img_dim))(data)
    data = data.reshape(len(data), -1)
    print("Data loaded")
    gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='full', verbose=2, verbose_interval=1)
    gmm.fit(data)

    # denoiser = GMMDenoiser(pi=torch.from_numpy(gmm.weights_),
    #                         mu=torch.from_numpy(gmm.means_),
    #                         sigma=torch.from_numpy(gmm.covariances_), device=torch.device('cpu'))

    joblib.dump(gmm, f"saved_models/GMM-k={n_components}.joblib")



def get_data(image_paths, grayscale=False, resize=None):
    """Read 'n_images' random images"""
    data = []
    for i, path in enumerate(image_paths):
        im = load_image(path, grayscale, resize=resize)
        data.append(im)
    data = torch.stack(data, dim=0) # [b, N, c*p*p]
    return data


if __name__ == '__main__':
    n_images = 5000
    data_path = '/cs/labs/yweiss/ariel1/data/FFHQ_128'
    path_list = [os.path.join(data_path, name) for name in json.load(open("../top_frontal_facing_FFHQ.txt", 'r'))[:n_images]]

    # data = get_data(path_list, grayscale=False, resize=None)
    # joblib.dump(data, f"saved_models/Frontal_FFHQ_N={n_images}.joblib")

    data = joblib.load("saved_models/Frontal_FFHQ_N=5000.joblib")
    train_low_res_GMM(data, 100, grayscale=False)