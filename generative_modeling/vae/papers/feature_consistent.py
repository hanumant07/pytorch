import torch
from typing import Tuple, List
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import models

Tensor = torch.Tensor


class VAEModel(nn.Module):
  """ Model architecture as described in arxiv.org/pdf/1610.00291.pdf".

    Deep Feature Consistent Variational Autoencoder.
    a) Upsample layer with nearest neighbor mode
    b) BN + LeakyRelu both encoder and decoder
    c) Latent space dim = 100
    d) Decoder stride = 1, kernel size = 3x3, padding = 1, replication
    """

  def __init__(self) -> None:
    super(VAEModel, self).__init__()
    self.encoder = nn.Sequential(
        nn.Conv2d(3, 32, 4, padding=1, padding_mode='replicate', stride=2),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(),
        nn.Conv2d(32, 64, 4, padding=1, padding_mode='replicate', stride=2),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
        nn.Conv2d(64, 128, 4, padding=1, padding_mode='replicate', stride=2),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
        nn.Conv2d(128, 256, 4, padding=1, padding_mode='replicate', stride=2),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(),
    )
    self.mu = nn.Linear(256 * 4 * 4, 100)
    self.log_var = nn.Linear(256 * 4 * 4, 100)
    self.decoder_fc = nn.Linear(100, 256 * 4 * 4)
    self.decoder = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(256, 128, 3, padding=1, padding_mode='replicate'),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(128, 64, 3, padding=1, padding_mode='replicate'),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(64, 32, 3, padding=1, padding_mode='replicate'),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(32, 3, 3, padding=1, padding_mode='replicate'),
    )

  def reparametrize(self, z_mu: Tensor, z_log_var: Tensor) -> Tensor:
    """ Implements reparametrization trick
    """
    std = torch.exp(z_log_var / 2)
    epsilon = torch.randn_like(std)
    z_out = z_mu + (epsilon * std)
    return z_out

  def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    x_encoded = self.encoder(x)
    x_encoded = x_encoded.view(-1, 256 * 4 * 4)
    mu = self.mu(x_encoded)
    log_var = self.log_var(x_encoded)

    z = self.reparametrize(mu, log_var)
    z = self.decoder_fc(z)
    z = z.view(-1, 256, 4, 4)
    reconstructed = self.decoder(z)
    return reconstructed, mu, log_var


class Trainer(object):
  """ Implements the training loop as described in arxiv.org/pdf/1610.00291.pdf

    a) Uses Adam Optimizer
    b) Implements the deep perceptual loss function and the KL Divergence as
       described in the paper
    c) Other hyperparams as described in the paper.

    Args:
      model: An instance of the VAE encoder as defined in the paper, or a
             matching forward pass vae
      vgg_variant: Either '123' or '345' as the 2 different VGG layer
                   combinations specified in the paper
      hyper_params: A dictionary containing the following key pair values
                    lr: default=0.005
                    num_epochs: default=5
                    batch_size: default=64
                    gamma: default=0.5
                    loss_alpha=1
                    loss_beta:0.5
                    Default values set as per paper
  """

  def __init__(self,
               model: nn.Module,
               vgg_variant: str,
               device: torch.device,
               hyper_params: dict = None) -> None:
    self._dev = device
    self._hp = {}
    if hyper_params is None:
      self._hp = {
          'lr': 0.005,
          'num_epochs': 5,
          'batch_size': 64,
          'gamma': 0.5,
          'loss_alpha': 1,
          'loss_beta': 0.5
      }
    else:
      self._hp = hyper_params

    if vgg_variant is None or vgg_variant != '123' or vgg_variant != '345':
      self._loss_variant = '123'
    else:
      self._loss_variant = '345'

    self._optim = optim.Adam(model.parameters(),
                             lr=self._hp['lr'],
                             weight_decay=self._hp['gamma'])
    self._loss_layers = self._get_vgg_layers()

  def _get_vgg_layers(self) -> List[nn.ModuleList]:
    """ Based on the loss variant downloads the vgg layers

    Return:
      Tuple containing slices of vgg19 corresponding to relu_*_*[1,2,3]
      or [3,4,5]
    """
    if self._loss_variant == '123':
      self._vgg_features_ind = [1, 3, 6]
    else:
      self._vgg_features_ind = [6, 8, 11]

    loss_layers = []
    prev_index = 0
    for i in self._vgg_features_ind:
      vgg_features = list(
          models.vgg19(pretrained=True).features[prev_index:i + 1])
      loss_layers.append(nn.ModuleList(vgg_features).eval().to(self._dev))
      prev_index = i + 1
    return loss_layers

  def _get_feature_perceptual_loss(self, recon: Tensor, orig: Tensor) -> Tensor:
    """ Computes the feature perceptual loss

    There are two variants '123' or '345' corresponding to the relu layer of
    vgg19 output respectively as per the particular variant

    Args:
      recon: batch of reconstructed images
      orig: batch of original images
    """
    total_loss = torch.as_tensor(0.0).to(self._dev)
    recon = recon.to(self._dev)
    orig = orig.to(self._dev)
    for i, module_list in enumerate(self._loss_layers):
      for layer in module_list:
        recon = layer.forward(recon)
        orig = layer.forward(orig)
      _, channels, width, height = orig.shape
      loss = (F.mse_loss(recon, orig,
                         reduction='sum')) / (channels * width * height)
      total_loss += loss

    return total_loss

  def _get_kl_loss(self, mu: Tensor, log_var: Tensor) -> Tensor:
    """ Compute the KL divergence

    Assuming N(0, 1) divergence is computed

    Args:
      mu: mean of Q(z|X)
      log_var: covariance of Q(z|X)

    Return:
      Tensor containing the kl loss
    """
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return kl

  def run_train_epoch(self, model: nn.Module, train_dl: DataLoader) -> float:
    """ Execute 1 epoch using the given dataloader

    Executes the training look for one epoch and returns the loss

    Args:
      model: The Vae model instance
      train_dl: Dataloader for the training dataset

    Returns:
      Loss for the epoch average of the epoch datasize
    """
    running_loss = 0.0
    running_size = 0
    for _, images, in enumerate(train_dl):
      original = images.to(self._dev)
      reconstructed, mu, log_var = model(original)
      perceptual_loss = self._get_feature_perceptual_loss(
          reconstructed, original)
      kl = self._get_kl_loss(mu, log_var)
      total_loss = self._hp['loss_alpha'] * kl + self._hp[
          'loss_beta'] * perceptual_loss
      self._optim.zero_grad()
      total_loss.backward()
      self._optim.step()
      running_loss += (total_loss.item())
      running_size += images.shape[0]

    return running_loss / running_size

  def run_test_loop(self, model: nn.Module, test_dl: DataLoader) -> float:
    """ Execute 1 iteration of the test set

    Returns the avg perceptual loss during the test loop
    Args:
      model: The Vae model instance
      test_dl: Dataloader for the test dataset

    Returns:
      Loss for the epoch average of the epoch datasize
    """
    running_loss = 0.0
    running_size = 0
    with torch.no_grad():
      for _, images in enumerate(test_dl):
        originals = images.to(self._dev)
        recon, _, _ = model(originals)
        perceptual_loss = self._get_feature_perceptual_loss(recon, originals)
        running_loss += (perceptual_loss.item())
        running_size += images.shape[0]
    return running_loss / running_size
