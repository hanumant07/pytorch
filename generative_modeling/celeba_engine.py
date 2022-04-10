from typing import Dict, Tuple
from absl import app
from absl import flags
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from vae.papers.feature_consistent import Trainer, VAEModel
from datasets.celeb_a import CelebADataset
from utils import ExperimentLogger

LEARNING_RATE = 0.01
BATCH_SIZE = 64
IMAGE_SIZE = 64
NUM_EPOCHS = 1
FLAGS = flags.FLAGS

flags.DEFINE_string('logs_path', '', help='Provide the path to save logs')
flags.DEFINE_string('data_path', '', help='Path to data directory')


def get_transforms() -> transforms.Compose:
  """ Returns transforms associated with celebA related papers

  Applies, resize->center crop->normalization
  """
  return transforms.Compose([
      transforms.Resize(IMAGE_SIZE),
      transforms.CenterCrop(IMAGE_SIZE),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])


def construct_hyperparams() -> Dict:
  """ Constructs dictionary of hyperparameters

  Hyperparameters defined in arxiv.org/pdf/1610.00291.pdf
  """
  return {
      'lr': 0.005,
      'num_epochs': NUM_EPOCHS,
      'batch_size': BATCH_SIZE,
      'gamma': 0.5,
      'loss_alpha': 1,
      'loss_beta': 0.5
  }


def get_device() -> torch.device:
  if torch.cuda.is_available():
    return torch.device('cuda')
  else:
    return torch.device('cpu')


def get_loaders() -> Tuple[DataLoader, DataLoader]:
  celeba_train_ds = CelebADataset(root=FLAGS.data_path,
                                  train=True,
                                  split=0.9,
                                  transforms=get_transforms())
  celeba_test_ds = CelebADataset(root=FLAGS.data_path,
                                 split=0.9,
                                 transforms=get_transforms())
  celeba_train_dl = DataLoader(celeba_train_ds,
                               shuffle=True,
                               batch_size=BATCH_SIZE)
  celeba_test_dl = DataLoader(celeba_test_ds, batch_size=BATCH_SIZE)
  return (celeba_train_dl, celeba_test_dl)


def main(argv):
  device = get_device()
  logger = ExperimentLogger(FLAGS.logs_path, 'celeba_vae')
  model = VAEModel().to(device)
  trainer = Trainer(model=model,
                    vgg_variant='123',
                    device=device,
                    hyper_params=construct_hyperparams())
  train_dl, test_dl = get_loaders()

  epoch_train_losses = []
  epoch_test_losses = []
  for epoch in range(NUM_EPOCHS):
    epoch_train_loss, train_loss_list = trainer.run_train_epoch(model, train_dl)
    epoch_test_loss, test_loss_list = trainer.run_test_loop(model, test_dl)
    epoch_train_losses.append(epoch_train_loss)
    epoch_test_losses.append(epoch_test_loss)
    logger.save_loss(train_loss_list, test_loss_list, epoch)
  logger.save_loss(epoch_train_losses, epoch_test_losses)


if __name__ == '__main__':
  app.run(main)
