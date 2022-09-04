from sklearn import datasets
import torch
from torch.utils.data import Dataset
import numpy as np

class DatasetMoons(Dataset):
  def __init__(self, cfg):
    super().__init__()

    self.cfg = cfg

  def __len__(self):
    return 1

  """ two half-moons """
  def __getitem__(self, idx):
    moon, labels = datasets.make_moons(n_samples=self.cfg.TRAINING.BATCH, noise=0.05)
    return torch.from_numpy(moon.astype(np.float32)), torch.from_numpy(labels)
