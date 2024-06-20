import torch
import torch.nn as nn
from torchvision import transforms

normal_root_dir = r'D:\PPJ\Model\data\Normal'
pre_data_dir = r'D:\PPJ\Model\pre_data'
root_dir = r'D\PPJ\Model\data'

height, width, channel = 480, 480, 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ngpu = 1
ksize = 4
z_dim = 20
learning_rate = 1e-3

img_size = 480

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

transform_pre = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

epochs = 100
batch_size = 32

avg_pool = nn.AdaptiveAvgPool2d((480, 480))