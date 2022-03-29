from typing import Tuple
import torch
import torch.nn as nn


Tensor = torch.Tensor

class CnnVAE(nn.Module):
    """ An implementation of the variational Autoencoder

    Uses CNN based encoder and decoder.
    """
    def __init__(self, num_channels: int):
        super(CnnVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 6, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 5, stride=2, padding=1,
                               output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 6, 5),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 1, 5, stride=2,
                               output_padding=1),
        )
        self.dist_mu = nn.Linear(64, 32)
        self.dist_log_var = nn.Linear(64, 32)
        self.dec_lin = nn.Linear(32, 64)

    def reparametrize(self, z_mu: Tensor, z_log_var: Tensor) -> Tensor:
        """ Implements reparametrization trick
        """
        std = torch.exp(z_log_var/2)
        epsilon = torch.randn_like(std)
        z_out = z_mu + (epsilon * std)
        return z_out

    def forward(self, batch_in: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """ Forward pass throught the encoder + decoder
        """
        x_encoded = self.encoder(batch_in)
        z_mu = self.dist_mu(x_encoded.view(-1, 64))
        z_log_var = self.dist_log_var(x_encoded.view(-1, 64))

        z_in = self.reparametrize(z_mu, z_log_var)

        z_in = self.dec_lin(z_in)
        z_in = z_in.view(-1, 64, 1, 1)
        recon = torch.sigmoid(self.decoder(z_in))
        return recon, z_mu, z_log_var
