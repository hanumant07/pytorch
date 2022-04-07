from typing import List
import matplotlib.pyplot as plt
import torch
import os.path
from torchvision.utils import save_image


class ExperimentLogger:
  """ Helps record the training artifacts for posterity

    Supports recording using tensorboard or to a specified output directory
    """

  def __init__(self, loc: str, exp_name: str = '') -> None:
    """ Create ImageUtils instance with a storage path for training artifacts

        Args:
          loc: string representing destination directory path
          exp_name: experiment specific prefix for saving artifacts

        Raises:
          ValueError: for bad location argument
        """
    if not os.path.isdir(loc):
      raise ValueError(f'Specified location, {loc} not a directory')
    self._location = loc
    self._prefix = exp_name

  @torch.no_grad()
  def save_recon(self, recon: torch.Tensor) -> None:
    """ Save reconstructed image grid to specified location

        Generally used with torchvision.utils.make_grid() output

        Args:
          recon: batch of reconstructed image tensors
    """
    output_file = os.path.join(self._loc, (self.exp_name + '_recon_images'))
    save_image(recon, f'{output_file}')

  @torch.no_grad()
  def save_loss(self, train_loss: List[float], val_loss: List[float]) -> None:
    """ Save the validation and taining loss
        """
    plt.figure(figsize=(20, 5))
    plt.xlabel('epoch')
    plt.ylable('Loss')
    plt.plot(train_loss, color='orange', label='train_loss')
    plt.plot(val_loss, color='blue', label='test_loss')
    plt.legend()
    plt.savefig(self._location)
    plt.show()
