import itertools
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms as tv_t
from tqdm import tqdm


def extract_intermediate_feature_maps(model, x, layer_indices):
    features = []
    for i, layer in enumerate(model):
        x = layer(x)
        if i in layer_indices:
            features.append(x)
    return features


def plot_img(axs, img, name):
    axs.imshow(np.clip(img.permute(1,2,0).cpu().numpy(), 0, 1), cmap='gray')
    axs.set_title(name, size=5)
    axs.axis('off')


def show_images(images_and_names, path=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, axs = plt.subplots(1, len(images_and_names), dpi=200, figsize=(2*len(images_and_names),2))
    for i, (image, name) in enumerate(images_and_names):
        plot_img(axs[i], image, name)
    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        plt.savefig(path)
    plt.clf()


def find_neural_nn(img, data, row_interval, col_interval, device):
    if img.shape[0] == 1:
        img = img.repeat(3,1,1)
    if row_interval is not None:
        img = img[:, row_interval[0]:row_interval[1], col_interval[0]: col_interval[1]]

    print("finding VGG NN")
    from torchvision import models, transforms
    vgg_features = models.vgg19(pretrained=True).features.to(device)
    vgg_features.eval()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    layers = [4, 9, 18] #, 27, 36]

    query_features = extract_intermediate_feature_maps(vgg_features, normalize(img).float(), layers)

    dists = []
    for i in tqdm(range(len(data))):
        ref = data[i]
        if row_interval is not None:
            ref = ref[:, row_interval[0]:row_interval[1], col_interval[0]: col_interval[1]]
        ref_features = extract_intermediate_feature_maps(vgg_features, normalize(ref).float(), layers)
        dist = np.sum([torch.norm(query_features[j] - ref_features[j]).item() for j in range(len(layers))])
        dists.append(dist)

    return np.argsort(dists)

def find_nn(img, data, row_interval, col_interval):
    if img.shape[0] == 1:
        img = img.repeat(3,1,1)
    if row_interval is not None:
        img = img[:, row_interval[0]:row_interval[1], col_interval[0]: col_interval[1]]

    dists = []
    for i in tqdm(range(len(data))):
        ref = data[i]
        if row_interval is not None:
            ref = ref[:, row_interval[0]:row_interval[1], col_interval[0]: col_interval[1]]
        dist = torch.norm(ref - img).item()
        dists.append(dist)

    return np.argsort(dists)

def find_crop_nns(query_image, raw_data):
    # Debug: show VGG NN
    img_dim = query_image.shape[-1]
    NN_images = []
    raw_data_subset = tv_t.Resize((img_dim, img_dim))(raw_data)
    d = img_dim // 4

    all_slices = np.arange(d,img_dim-d, d)
    all_slices = list(zip(all_slices, all_slices + d))
    all_pairs_of_slices = itertools.product(all_slices, repeat=2)
    for i, (row_interval, col_interval) in enumerate(all_pairs_of_slices):
        nn_indices = find_nn(query_image, raw_data_subset, row_interval, col_interval)
        # nn_indices = find_neural_nn(query_image, raw_data_subset, row_interval, col_interval, raw_data.device)
        nn = raw_data_subset[nn_indices[0]].clone() * 0.5
        nn[:, row_interval[0]:row_interval[1], col_interval[0]: col_interval[1]] *= 2
        NN_images.append((nn, f'crop vgg {(row_interval, col_interval)} -NN'))

    return NN_images