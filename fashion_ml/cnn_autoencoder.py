import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_channel, middle_channel, code_channel, kernel_size, code_size, encoder_depth):
        super().__init__()
        encoder_layers = [
            nn.Conv2d(input_channel, middle_channel, kernel_size),
            nn.ReLU()
        ]
        
        for i in range(encoder_depth):
            encoder_layers.append(nn.Conv2d(middle_channel, middle_channel, kernel_size))
            encoder_layers.append(nn.ReLU())

        encoder_layers.append(nn.Conv2d(middle_channel, code_channel, kernel_size))
        encoder_layers.append(nn.Sigmoid())

        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = [
            nn.Conv2d(code_channel, middle_channel, kernel_size),
            nn.ReLU()
        ]

        for i in range(encoder_depth):
            decoder_layers.append(nn.Conv2d(middle_channel, middle_channel))
            decoder_layers.append(nn.ReLU())

        decoder_layers.append(nn.Conv2d(middle_channel, input_channel))
        decoder_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder_layers)


    def forward(self, x):
        code = self.encoder(x)
        x_prime = self.decoder(code)

        return x_prime
