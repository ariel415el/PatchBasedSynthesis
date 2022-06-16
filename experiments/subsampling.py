import torchvision

from utils import load_image, get_patches
from torchvision.utils import save_image


def main():
    p = 16
    s = 32
    image = load_image('/mnt/storage_ssd/datasets/FFHQ_128/69990.png')

    patches = get_patches(image, p, s).reshape(-1,3,p,p)

    patch = patches[6:7]
    resized_patch = torchvision.transforms.Resize((8,8), torchvision.transforms.InterpolationMode.NEAREST)(patch)

    save_image(patch, "patch.png", normalize=True)
    save_image(resized_patch, "resized_patch.png", normalize=True)

if __name__ == '__main__':
    main()