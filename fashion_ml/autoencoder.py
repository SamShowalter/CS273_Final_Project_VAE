import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_size, middle_size, code_size, encoder_depth):
        super().__init__()
        encoder_layers = [
            nn.Linear(input_size, middle_size),
            nn.ReLU()
        ]
        
        for i in range(encoder_depth):
            encoder_layers.append(nn.Linear(middle_size, middle_size))
            encoder_layers.append(nn.ReLU())

        encoder_layers.append(nn.Linear(middle_size, code_size))
        encoder_layers.append(nn.Sigmoid())

        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = [
            nn.Linear(code_size, middle_size),
            nn.ReLU()
        ]

        for i in range(encoder_depth):
            decoder_layers.append(nn.Linear(middle_size, middle_size))
            decoder_layers.append(nn.ReLU())

        decoder_layers.append(nn.Linear(middle_size, input_size))
        decoder_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder_layers)


    def forward(self, x):
        code = self.encoder(x)
        x_prime = self.decoder(code)

        return x_prime
    
if __name__ == "__main__":
    ae = AutoEncoder(10, 5, 2, 2)
    print(ae(torch.tensor([[1.0]*10], dtype=torch.float)))

