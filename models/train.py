import os
import sys
import torch

import joblib

sys.path.append(os.path.dirname(__file__))
from utils import load_image
import json


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

    data = get_data(path_list, grayscale=False, resize=None)
    joblib.dump(data, f"saved_models/Frontal_FFHQ_N={n_images}.joblib")
