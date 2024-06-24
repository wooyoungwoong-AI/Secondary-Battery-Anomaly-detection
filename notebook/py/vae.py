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
                    ngpu=self.ngpu, ksize=self.ksize, z_dim=self.z_dim).to(self.device)
        )

        self.decoder = (
            Decoder(height=self.height, width=self.width, channel=self.channel,
                    ngpu=self.ngpu, ksize=self.ksize, z_dim=self.z_dim).to(self.device)
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
        # self.optimizer = optim.Adam(self.params, lr=self.learning_rate)
    
    def to(self, device):
        for idx_m, model in enumerate(self.models):
            self.models[idx_m] = model.to(device)
        self.device = device

        self.optimizer = optim.Adam(self.params, lr=self.learning_rate)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def train(self, mode=True):
        for model in self.models:
            model.train(mode)

    
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
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

        self.conv_mu = nn.Conv2d(64, z_dim, kernel_size=1)
        self.conv_sigma = nn.Conv2d(64, z_dim, kernel_size=1)

        # self.encoder_dense = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(123 * 123 * 64, 512),
        #     nn.ELU(),
        #     nn.Linear(512, self.z_dim*2)
        # )

    # def split_z(self, z):
    #     z_mu = z[:, :self.z_dim]
    #     z_sigma = z[:, self.z_dim:]

    #     return z_mu, z_sigma
    
    # #reparameterzation
    # def sample_z(self, mu, sigma):
    #     epsilon = torch.randn_like(mu)
    #     sample = mu + (sigma * epsilon)

    #     return sample
    
    def forward(self, input):
        conv_out = self.encoder_conv(input)
        mu = self.conv_mu(conv_out)
        sigma = torch.exp(0.5 * self.conv_sigma(conv_out))
        epsilon = torch.randn_like(sigma)
        z_sample = mu + sigma * epsilon

        return z_sample, mu, sigma
    
class Decoder(nn.Module):
    def __init__(self, height, width, channel, ngpu, ksize, z_dim):
        super(Decoder, self).__init__()

        self.height, self.width, self.channel = height, width, channel
        self.ngpu, self.ksize, self.z_dim = ngpu, ksize, z_dim

        # self.decoder_dense = nn.Sequential(
        #     nn.Linear(self.z_dim, 512),
        #     nn.ELU(),
        #     nn.Linear(512, (self.height//(2**2))*(self.width//(2**2))*64),
        #     nn.ELU()
        # )

        self.decoder_dense = nn.Sequential(
            nn.Conv2d(in_channels=z_dim, out_channels=64, kernel_size=1),
            nn.ELU()
        )

        self.decoder_conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.ksize+1, stride=2, padding=1, output_padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=self.ksize+1, stride=2, padding=1, output_padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=self.channel, kernel_size=self.ksize, stride=1, padding=self.ksize//2),
            # nn.AdaptiveAvgPool2d((height, width)),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        dense_out = self.decoder_dense(input)
        x_hat = self.decoder_conv(dense_out)
        return x_hat