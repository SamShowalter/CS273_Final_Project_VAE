import torch
import torch.nn as nn
from numpy import prod

class AutoEncoder(nn.Module):
    def __init__(self, input_channel = 1,
                 input_dim = 28*28,
                 hdims = [32, 32, 64],
                 strides = [1,2,2],
                 paddings = [2,1,1],
                 final_shape = (64,7,7),
                 latent_dim = 3,
                 kernel_size = 4):
        super().__init__()
        
        encoder_layers = []
        ip = input_channel
        for dim, stride, pad in zip(hdims, strides, paddings):
            encoder_layers.append(nn.Conv2d(ip, dim, kernel_size=kernel_size, stride=stride, padding=pad))
            encoder_layers.append(nn.ReLU())
            ip = dim

        encoder_layers.append(nn.Flatten())
        encoder_layers.append(nn.Linear(prod(final_shape), latent_dim))
        encoder_layers.append(nn.Sigmoid())

        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = [
            nn.Linear(latent_dim, prod(final_shape)),
            nn.Unflatten(1, final_shape)
        ]
        ip = hdims[-1]
        for i, (dim, stride, pad) in reversed(list(enumerate(zip([input_channel] + hdims[:-1], strides, paddings)))):
            decoder_layers.append(nn.ConvTranspose2d(ip, dim, kernel_size=kernel_size, stride=stride, padding=pad, output_padding = 1 if i == 1 else 0))
            if i == 0:
                decoder_layers.append(nn.Sigmoid())
            else:
                decoder_layers.append(nn.ReLU())
            ip = dim

        # decoder_layers.append(nn.Conv2d(ip, input_channel, kernel_size))
        # decoder_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder_layers)


    def forward(self, x):
        code = self.encoder(x)
        x_prime = self.decoder(code)

        return x_prime
