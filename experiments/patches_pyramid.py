import numpy as np
import torchvision
from torchvision.utils import save_image

from utils import load_image, get_patches


def main():
    p = 32
    s = 32
    n_windows_per_dim = 4
    image = load_image('/cs/labs/yweiss/ariel1/data/FFHQ_128/69999.png')
    h,w = image.shape[-2:]

    patches = get_patches(image, p, s).reshape(-1,3,p,p)

    save_image(patches, f"patches.png", normalize=True, nrow=int(np.sqrt(len(patches))))

    patches = torchvision.transforms.Resize((p // 2, p // 2))(patches)
    save_image(patches, f"downsized_2.png", normalize=True, nrow=int(np.sqrt(len(patches))))
    patches = torchvision.transforms.Resize((p // 4, p // 4))(patches)
    save_image(patches, f"downsized_4.png", normalize=True, nrow=int(np.sqrt(len(patches))))
    patches = torchvision.transforms.Resize((p // 8, p // 8))(patches)
    save_image(patches, f"downsized_8.png", normalize=True, nrow=int(np.sqrt(len(patches))))



if __name__ == '__main__':
    main()