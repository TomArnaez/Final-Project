from torchvision import datasets
import numpy as np
import os
import PIL
import torch
import torchvision

from PIL import Image
from torch.utils.data import Subset

PERMUTATIONS = [
    "brightness",
    "gaussian_blur",
    "gaussian_noise",
    "gaussian_noise_2",
    "gaussian_noise_3",
    "motion_blur",
    "rotate",
    "scale",
    "shear",
    "shot_noise",
    "shot_noise_2",
    "shot_noise_3",
    "snow",
    "spatter",
    "speckle_noise",
    "speckle_noise_2",
    "speckle_noise_3",
    "tilt",
    "translate",
    "zoom_blur"
]

CORRUPTIONS = [
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "natural",
    "motion_blur",
    "pixelate",
    "shot_noise",
    "snow",
    "zoom_blur"
]

class CIFAR10P(datasets.VisionDataset):
    def __init__(self, root :str, name :str,
                 transform=None):
        assert name in PERMUTATIONS
        super(CIFAR10P, self).__init__(
            root, transform=transform
        )
        data_path = os.path.join(root, name + '.npy')
        
        self.data = np.load(data_path)
        
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def __len__(self):
        return len(self.data)

class CIFAR10C(datasets.VisionDataset):
    def __init__(self, root :str, name :str,
                 transform=None, target_transform=None):
        assert name in CORRUPTIONS
        super(CIFAR10C, self).__init__(
            root, transform=transform,
            target_transform=target_transform
        )
        data_path = os.path.join(root, name + '.npy')
        target_path = os.path.join(root, 'labels.npy')
        
        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
            
        return img, targets
    
    def __len__(self):
        return len(self.data)
    
def extract_subset(dataset, num_subset :int, random_subset :bool):
    if random_subset:
        random.seed(0)
        indices = random.sample(list(range(len(dataset))), num_subset)
    else:
        indices = [i for i in range(num_subset)]
    return Subset(dataset, indices)