import cv2
import torch
import torchvision
from torch.nn import functional as F
import torchvision.transforms as tv_t


def load_image(im_path, grayscale=False, resize=None):
    im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB) / 255.
    im = torch.from_numpy(im).permute(2,0,1)
    if resize is not None:
        im = tv_t.Resize((resize, resize))(im)
    if grayscale:
        im = torch.mean(im, dim=0, keepdim=True)
    return im


def get_patches(x, p, stride):
    flat_x = F.unfold(x.unsqueeze(0), kernel_size=p, stride=stride)[0]  # [c*p*p, N]
    return flat_x.T # .reshape(flat_x.shape[1], -1, p, p)


def resize_img(x, dim):
    return torchvision.transforms.Resize(dim, interpolation=torchvision.transforms.InterpolationMode.NEAREST)(x)


def get_derivative(data, c):
    grayscale = torch.mean(data, dim=1, keepdims=True)
    laplace = torch.tensor([[0, 1, 0],[1, -4, 1], [0, 1, 0]]).unsqueeze(0).unsqueeze(0).cuda().float()
    return F.conv2d(grayscale, laplace, padding='same').repeat(1,c, 1, 1)