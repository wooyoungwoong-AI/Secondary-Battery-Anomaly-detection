import os
import matplotlib.pyplot as plt
plt.rc('font', family='NanumGothic')
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
from tqdm import tqdm

import py.config as CFG

folder = CFG.root_folder
data_path = []
for filename in os.listdir(folder):
    data_path.append(os.path.join(folder, filename))

# # topLeft center topRight bottomLeft bottomRight
# sin_ticket_list = [
#     [40, 100, 880, 940],
#     [810, 100, 1650, 940],
#     [1580, 100, 2420, 940],
#     [30, 880, 870, 1720],
#     [1590, 880, 2430, 1720]
# ]
# # image size (840, 840)
# sin_ticket_list_error = [
#     [10, 10, 850, 850],
#     [780, 30, 1620, 870],
#     [1560, 40, 2400, 880],
#     [10, 780, 850, 1620],
#     [1550, 800, 2390, 1640]
# ]
# sinner = 0
# weare = 770

def rotate_images(image):
    images = []
    for angle in [0, 90, 180, 270]:
        rotated_image = TF.rotate(image, angle)
        images.append(rotated_image)
    return images

def gauss_noise(image_tensor, sigma=0.05):
    noise = torch.randn(image_tensor.size()) * sigma
    noisy_image = image_tensor + noise
    noisy_image = torch.clamp(noisy_image, 0, 1)
    return noisy_image

class VAECustomDataset(Dataset):
    def __init__(self, file_paths, transform=None, gauss_sigma=0.05):
        self.file_paths = file_paths
        self.transform = transform
        self.gauss_sigma = gauss_sigma

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = Image.open(self.file_paths[idx])
        if self.transform:
            original_image = self.transform(image)
            noisy_image = gauss_noise(original_image, self.gauss_sigma)
            images = rotate_images(image)
            transformed_images = [self.transform(img) for img in images]
            noisy_images = [gauss_noise(img, self.gauss_sigma) for img in transformed_images]
            return original_image, noisy_image, transformed_images, noisy_images
        else:
            return image

transform = CFG.transform