import os, time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc

from torch.utils.tensorboard import SummaryWriter


def make_dir(path):

    if not os.path.exists(path):
        os.makedirs(path)

def create_makedirs():
    result_dir = 'results'
    make_dir(result_dir)
    #tr_latent : latent space에 암축된 이미지 시각화한 이미지를 저장
    #tr_restoring : 원본 이미지와 복원 이미지의 차이를 시각화한 이미지를 저장
    #tr_latent_walk : latent space에 따라 생성된 이미지를 저장
    subdirs = ['tr_latent', 'tr_restoring', 'tr_latent_walk']
    for subdir in subdirs:
        make_dir(os.path.join(result_dir, subdir))

def initialize_training_variables():
    # start_time: 학습 시작 시간을 기록하여 전체 학습 시간을 계산합니다
    # iteration: 현재까지의 반복 횟수를 기록
    # writer: TensorBoard에 로그를 작성하기 위한 객체
    start_time = time.time()
    iteration = 0
    writer = SummaryWriter()
    return start_time, iteration, writer

def prepare_data_loader(dataset, batch_size, shuffle=True):
    train_loder = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loder

def loss_function(x, x_hat, mu, sigma):
    x, x_hat, mu, sigma = x.cpu(), x_hat.cpu(), mu.cpu(), sigma.cpu()
    #복원오류 계산
    restore_error = -torch.sum(x * torch.log(x_hat + 1e-12) + (1 - x) * torch.log(1 - x_hat + 1e-12), dim=(1, 2, 3))
    #KL Divergence 계산
    kl_divergence = 0.5 * torch.sum(mu**2 + sigma**2 - torch.log(sigma**2 + 1e-12) - 1, dim=(1))
    
    return torch.mean(restore_error + kl_divergence), torch.mean(restore_error), torch.mean(kl_divergence)

def train_epoch(epoch, neuralnet, train_loader, writer, iteration):
    neuralnet.train()
    list_recon, list_kld, list_total = [], [], []

    for batch_idx, (original_image, noisy_image, transformed_images, noisy_images) in enumerate(train_loader):
        original_image = original_image.to(neuralnet.device)
        noisy_image = noisy_image.to(neuralnet.device)

        z_enc, z_mu, z_sigma = neuralnet.encoder(noisy_image)
        x_hat = neuralnet.decoder(z_enc)

        total_loss, restore_error, kl_divergence = loss_function(
            x = original_image, x_hat=x_hat, mu=z_mu, sigma=z_sigma
        )

        neuralnet.optimizer.zero_grad()
        total_loss.backward()
        neuralnet.step()

        list_recon.append(restore_error.item())
        list_kld.append(kl_divergence.item())
        list_total.append(total_loss.item())

        #TensorBoard logging
        writer.add_scalar('VAE/restore_error', restore_error, iteration)
        writer.add_scalar('VAE/kl_divergence', kl_divergence, iteration)
        writer.add_scalar('VAE/totla_loss', total_loss, iteration)

        iteration += 1

        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch+1}], Step [{batch_idx+1}/{len(train_loader)}]',
                  f'Restore Error : {restore_error.item():.4f}, KLD : {kl_divergence.item():.4f}, Total Loss : {total_loss.item():.4f}')
    
    return list_recon, list_kld, list_total, iteration