import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNet:
    def __init__(self, height, width, channel, device, ngpu, ksize, z_dim, learning_rate=1e-3):
        self.height, self.width, self.channel = height, width, channel
        self.device, self.ngpu = device, ngpu
        self.ksize, self.z_dim, self.learning_rate = ksize, z_dim, learning_rate

        self.encoder = (
            Encoder(height=self.height, width=self.width, channel=self.channel,
                    ngpu=self.ngpu, z_dim=self.z_dim).to(self.device)
        )
    
class Flatten(nn.Module):
    def forward(self, input):
        flatten = nn.Flatten()
        return flatten(input)
    
class Encoder(nn.Module):
    def __init__(self, height, width, channel, ngpu, ksize, z_dim):
        super(Encoder, self).__init__()

        self.height, self.width, self.channel = height, width, channel
        self.ngpu, self.ksize, self.z_dim = ngpu, ksize, z_dim

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.channel, out_channels=16, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.ELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.ELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.ELU(),
        )

        self.encoder_dense = nn.Sequential(
            Flatten(),
            nn.Linear((self.height//(2**2))*(self.width//(2**2))*self.channel*64, 512),
            nn.ELU(),
            nn.Linear(512, self.z_dim*2)
        )

    def split_z(self, z):
        z_mu = z[:, :self.z_dim]
        z_sigma = z[:, self.z_dim:]

        return z_mu, z_sigma
    
    def sample_z(self, mu, sigma):
        epsilon = torch.randn_like(mu)
        sample = mu + (sigma * epsilon)

        return sample
    
    def forward(self, input):
        conv_out = self.encoder_conv(input)
        z_params = self.encoder_dense(conv_out)
        z_mu, z_sigma = self.split_z(z_params)
        z_sample = self.sample_z(mu=z_mu, sigma=z_sigma)

        return z_sample, z_mu, z_sigma
    
class Decoder(nn.Module):
    def __init__(self, height, width, channel, ngpu, ksize, z_dim):
        super(Decoder, self).__init__()

        self.height, self.width, self.channel = height, width, channel
        self.ngpu, self.ksize, self.z_dim = ngpu, ksize, z_dim

        self.decoder_dense = nn.Sequential(
            nn.Linear(self.z_dim, 512),
            nn.ELU(),
            nn.Linear(512, (self.height//(2**2))*(self.width//(2**2))*self.channel*64),
            nn.ELU()
        )

        self.decoder_conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.ELU(),
        )