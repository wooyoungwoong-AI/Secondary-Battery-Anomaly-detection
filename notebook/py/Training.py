import torch
from torch.nn import functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc

from torch.utils.tensorboard import SummaryWriter

def loss_function(x, x_hat, mu, sigma):
    x, x_hat, mu, sigma = x.cpu(), x_hat.cpu(), mu.cpu(), sigma.cpu()
    #복원오류 계산
    restore_error = -torch.sum(x * torch.log(x_hat + 1e-12) + (1 - x) * torch.log(1 - x_hat + 1e-12), dim=(1, 2, 3))
    #KL Divergence 계산
    kl_divergence = 0.5 * torch.sum(mu**2 + sigma**2 - torch.log(sigma**2 + 1e-12) - 1, dim=(1))
    
    return torch.mean(restore_error + kl_divergence), torch.mean(restore_error), torch.mean(kl_divergence)

def training(neuralnet, dataset, epochs, batch_size):
