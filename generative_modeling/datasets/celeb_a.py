import torchvision
import torch
from typing import List
from torch.utils.data import Dataset
from PIL import Image
import os


class CelebADataset(Dataset):
  """ An implementation of the CelebA dataset

    Requires the dataset to be downloaded previously and unzipped

    Args:
      root:  The base directory containg the downloaded dataset.
      split: A value between 0 and 1 representing the data split size for given
             instance
      transforms: Transforms to apply to a sample prior to returning it
  """

  def __init__(self,
               root: str,
               train: bool = False,
               split: float = 0.8,
               transforms: torchvision.transforms = None) -> None:
    self._root = root
    if not os.path.isdir(self._root):
      return None
    img_list = self._get_celeb_list(root)
    total_len = len(img_list)
    train_len = int(split * total_len)

    test_len = total_len - train_len
    if train is True:
      self._celeb_img_list = img_list[:train_len]
    else:
      self._celeb_img_list = img_list[-test_len:]
    self._transforms = transforms

  def _get_celeb_list(self, base_dir: str) -> List[str]:
    """ From the download root, retrieven the paths for different images
    """
    image_dir = os.path.join(base_dir, 'img_align_celeba')
    with open(os.path.join(base_dir, 'identity_CelebA.txt')) as f:
      lines = f.readlines()
      result = []
      for name, _ in (line.split(' ') for line in lines):
        result.append(os.path.join(image_dir, name))
      return sorted(result)

  def __len__(self):
    return len(self._celeb_img_list)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.toList()
    sample = Image.open(self._celeb_img_list[idx]).convert('RGB')
    if self._transforms:
      sample = self._transforms(sample)
    return sample
