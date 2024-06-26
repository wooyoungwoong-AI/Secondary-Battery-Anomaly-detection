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
    plt.savefig(f"{savename}.png")
    plt.close()

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_makedirs():
    result_dir = 'results'
    make_dir(result_dir)
    subdirs = ['tr_latent', 'tr_restoring', 'tr_latent_walk']
    for subdir in subdirs:
        make_dir(os.path.join(result_dir, subdir))

def initialize_training_variables():
    start_time = time.time()
    iteration = 0
    writer = SummaryWriter()
    return start_time, iteration, writer

def prepare_data_loaders(dataset, batch_size, shuffle=True):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

def loss_function(x, x_hat):
    x, x_hat = x.cpu(), x_hat.cpu()
    restore_error = torch.nn.functional.mse_loss(x_hat, x, reduction='mean')
    return restore_error

def train_epoch(epoch, neuralnet, train_loader, writer, iteration):
    neuralnet.encoder.train()
    neuralnet.decoder.train()
    list_total = []

    for batch_idx, images in enumerate(train_loader):
        original_image = images[0].to(neuralnet.device)
        noisy_image = images[1].to(neuralnet.device)

        conv_out = neuralnet.encoder(noisy_image)
        x_hat = neuralnet.decoder(conv_out)

        total_loss = loss_function(x=original_image, x_hat=x_hat)

        neuralnet.optimizer.zero_grad()
        total_loss.backward()
        neuralnet.optimizer.step()

        list_total.append(total_loss.item())

        writer.add_scalar('VAE/total_loss', total_loss, iteration)
        iteration += 1

        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch+1}], Step [{batch_idx+1}/{len(train_loader)}], '
                  f'Total Loss: {total_loss.item()}')

    return list_total, iteration

def test_epoch(neuralnet, test_loader):
    neuralnet.encoder.eval()
    neuralnet.decoder.eval()
    total_loss = 0

    with torch.no_grad():
        for images in test_loader:
            original_image = images[0].to(neuralnet.device)
            noisy_image = images[1].to(neuralnet.device)

            conv_out = neuralnet.encoder(noisy_image)
            x_hat = neuralnet.decoder(conv_out)

            tot_loss = loss_function(x=original_image, x_hat=x_hat)
            total_loss += tot_loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f"Test Results - Average Loss: {avg_loss}")

    return avg_loss

def save_test_results(avg_loss, filename="test_results.txt"):
    with open(filename, 'w') as f:
        f.write(f"Average Loss: {avg_loss}\n")

def training_and_testing(neuralnet, train_dataset, test_dataset, epochs, batch_size):
    create_makedirs()
    start_time, iteration, writer = initialize_training_variables()
    train_loader = prepare_data_loaders(train_dataset, batch_size)
    test_loader = prepare_data_loaders(test_dataset, batch_size, shuffle=False)

    for epoch in range(epochs):
        list_total, iteration = train_epoch(
            epoch, neuralnet, train_loader, writer, iteration
        )

        avg_epoch_loss = sum(list_total) / len(list_total)
        print(f"Epoch [{epoch+1}/{epochs}], Total Iteration: {iteration}, "
              f"Total Loss: {avg_epoch_loss:.4f}")

        # 모델의 가중치 저장
        torch.save(neuralnet.encoder.state_dict(), f"results/encoder_epoch_{epoch+1}.pth")
        torch.save(neuralnet.decoder.state_dict(), f"results/decoder_epoch_{epoch+1}.pth")

    avg_loss = test_epoch(neuralnet, test_loader)
    save_test_results(avg_loss)

    elapsed_time = time.time() - start_time
    print(f"Training and testing complete. Elapsed time: {elapsed_time:.2f} seconds")

    save_graph(contents=list_total, xlabel="Iteration", ylabel="Total Loss", savename="loss_total")
