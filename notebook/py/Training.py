import os, time

import numpy as np

import torch
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt

from torch.utils.tensorboard import SummaryWriter


def save_graph(contents, xlabel, ylabel, savename):

    np.save(savename, np.asarray(contents))
    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.plot(contents, color='blue', linestyle="-", label="loss")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.savefig("%s.png" %(savename))
    plt.close()

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

def prepare_data_loaders(dataset, batch_size, shuffle=True):
    train_loder = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loder

#train
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

        #forward pass
        z_enc, z_mu, z_sigma = neuralnet.encoder(noisy_image)
        x_hat = neuralnet.decoder(z_enc)

        #calculate loss
        total_loss, restore_error, kl_divergence = loss_function(
            x = original_image, x_hat=x_hat, mu=z_mu, sigma=z_sigma
        )

        neuralnet.optimizer.zero_grad()
        total_loss.backward()
        neuralnet.step()

        #record error
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

#test
def prepare_test_loader(dataset, batch_size, shuffle=False):
    test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return test_loader

def set_model_to_eval_model(model):
    model.eval()

def test_epoch(neuralnet, test_loader):
    neuralnet.eval()
    total_loss, total_kl_divergence, total_restore_error = 0, 0, 0

    with torch.no_grad():
        for origianl_image, noisy_image, transformed_images, noisy_image in test_loader:
            origianl_image = origianl_image.to(neuralnet.device)
            noisy_image = noisy_image.to(neuralnet.device)

            #forward pass
            z_enc, z_mu, z_sigma = neuralnet.encoder(noisy_image)
            x_hat = neuralnet.decoder(z_enc)

            #calculate loss
            tot_loss, restore_error, kl_divergence = loss_function(
                x=origianl_image, x_hat=x_hat, mu=z_mu, sigma=z_sigma
            )

            total_loss += tot_loss.item()
            total_restore_error += restore_error.item()
            total_kl_divergence += kl_divergence.item()

    avg_loss = total_loss / len(test_loader)
    avg_restore_error = total_restore_error / len(test_loader)
    avg_kl_divergence = total_kl_divergence / len(test_loader)

    print(f"Test Results - Average Loss: {avg_loss:.4f}, Average Restore Error: {avg_restore_error:.4f}, Average KL Divergence: {avg_kl_divergence:.4f}")

    return avg_loss, avg_restore_error, avg_kl_divergence

def save_test_results(avg_restore_error, avg_kl_divergence, avg_loss, filename="test_results.txt"):
    with open(filename, 'w') as f:
        f.write(f"Average Restore Error: {avg_restore_error:.4f}\n")
        f.write(f"Average KL Divergence: {avg_kl_divergence:.4f}\n")
        f.write(f"Average Loss: {avg_loss:.4f}\n")

def training_and_testing(neuralnet, train_dataset, test_dataset, epochs, batch_size):
    create_makedirs()
    start_time, writer, iteration = initialize_training_variables()
    train_loader = prepare_data_loaders(train_dataset, batch_size)
    test_loader = prepare_test_loader(test_dataset, batch_size)

    for epoch in range(epochs):
        list_recon, list_kld, list_total, iteration = train_epoch(
            epoch, neuralnet, train_loader, writer, iteration
        )

        print(f"Epoch [{epoch+1}/{epochs}], Total Iteration: {iteration}, "
              f"Restore Error: {sum(list_recon)/len(list_recon):.4f}, "
              f"KLD: {sum(list_kld)/len(list_kld):.4f}, "
              f"Total Loss: {sum(list_total)/len(list_total):.4f}")

        #model save
        for idx_m, model in enumerate(neuralnet.models):
            torch.save(model.state_dict(), f"results/params-{idx_m}.pth")

    #test
    avg_loss, avg_restore_error, avg_kl_divergence = test_epoch(neuralnet, test_loader)
    save_test_results(avg_restore_error, avg_kl_divergence, avg_loss)

    elapsed_time = time.time() - start_time
    print(f"Training and testing complete. Elapsed time: {elapsed_time:.2f} seconds")

    save_graph(contents=list_recon, xlabel="Iteration", ylabel="Reconstruction Error", savename="restore_error")
    save_graph(contents=list_kld, xlabel="Iteration", ylabel="KL-Divergence", savename="kl_divergence")
    save_graph(contents=list_total, xlabel="Iteration", ylabel="Total Loss", savename="loss_total")
