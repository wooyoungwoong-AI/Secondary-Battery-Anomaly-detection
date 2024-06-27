import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class NeuralNet:
    def __init__(self, height, width, channel, device, ngpu, learning_rate=1e-3):
        self.height, self.width, self.channel = height, width, channel
        self.device, self.ngpu = device, ngpu
        self.learning_rate = learning_rate

        self.encoder = (
            Encoder(height=self.height, width=self.width, channel=self.channel,
                    ngpu=self.ngpu).to(self.device)
        )

        self.decoder = (
            Decoder(height=self.height, width=self.width, channel=self.channel,
                    ngpu=self.ngpu).to(self.device)
        )

        self.models = [self.encoder, self.decoder]

        # GPU 설정
        for idx_m, model in enumerate(self.models):
            if self.device.type == 'cuda' and getattr(model, 'ngpu', 0) > 0:
                self.models[idx_m] = nn.DataParallel(model, list(range(model.ngpu)))

        # 파라미터 수 출력
        self.num_params = 0
        for idx_m, model in enumerate(self.models):
            for p in model.parameters():
                self.num_params += p.numel()
        print(f"The number of parameters: {self.num_params:.4f}")

        self.params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.Adam(self.params, lr=self.learning_rate)
    
    def to(self, device):
        for idx_m, model in enumerate(self.models):
            self.models[idx_m] = model.to(device)
        self.device = device

    def train(self, mode=True):
        for model in self.models:
            model.train(mode)

    
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class Encoder(nn.Module):
    def __init__(self, height, width, channel, ngpu):
        super(Encoder, self).__init__()

        self.height, self.width, self.channel = height, width, channel
        self.ngpu = ngpu

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.channel, out_channels=8, kernel_size=3, stride=1, padding='same'),
            nn.ELU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding='same'),
            nn.ELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding='same'),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding='same'),
            nn.ELU(),
            # nn.MaxPool2d(4),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding='same'),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.ELU(),
            # nn.MaxPool2d(4),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.ELU(),
            # nn.MaxPool2d(3),
        )

        # self.conv_mu = nn.Conv2d(16, 16, kernel_size=1)
        # self.conv_sigma = nn.Conv2d(16, 16, kernel_size=1)
    
    def forward(self, input):
        conv_out = self.encoder_conv(input)
        # mu = self.conv_mu(conv_out)
        # sigma = torch.exp(0.5 * self.conv_sigma(conv_out))
        # epsilon = torch.randn_like(sigma)
        # z_sample = mu + sigma * epsilon

        return conv_out

class Decoder(nn.Module):
    def __init__(self, height, width, channel, ngpu):
        super(Decoder, self).__init__()

        self.height, self.width, self.channel = height, width, channel
        self.ngpu = ngpu

        self.decoder_dense = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
            nn.ELU()
        )

        self.decoder_conv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.ELU(),
            # nn.Upsample(scale_factor=3, mode='nearest'),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding='same'),
            nn.ELU(),
            # nn.Upsample(scale_factor=4, mode='nearest'),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding='same'),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding='same'),
            nn.ELU(),
            # nn.Upsample(scale_factor=4, mode='nearest'),

            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding='same'),
            nn.ELU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding='same'),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding='same'),
            nn.ELU(),
            nn.Conv2d(in_channels=8, out_channels=self.channel, kernel_size=3, stride=1, padding='same'),
            nn.ELU(),
        )

    def forward(self, input):
        dense_out = self.decoder_dense(input)
        x_hat = self.decoder_conv(dense_out)
        return x_hat
