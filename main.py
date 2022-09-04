import os
import random
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sys import exit as e
from icecream import ic

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from model import Glow
from dataset import DatasetMoons
from loss import CustomLoss
import util as ut
from configs import get_cfg_defaults


def seed_everything(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True

def get_args():
  parser = argparse.ArgumentParser(description="Vision Transformers")
  parser.add_argument('--config', type=str, default='default', help='configuration to load')
  args = parser.parse_args()
  return args

def my_collate(batch):
  x, target = batch[0]
  ind_0 = (target == 0).nonzero(as_tuple=True)[0]
  ind_1 = (target == 1).nonzero(as_tuple=True)[0]
  return x, target, (ind_0, ind_1)

if __name__ == "__main__":
  seed_everything(42)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  torch.autograd.set_detect_anomaly(True)

  print("GPU: ", torch.cuda.is_available())
  args = get_args()
  config_path = os.path.join("./configs/experiments", f"{args.config}.yaml")

  cfg = get_cfg_defaults()
  cfg.merge_from_file(config_path)
  cfg.freeze()
  print(cfg)

  model = Glow(cfg)
  # d = DatasetMoons()
  dataset = DatasetMoons(cfg)
  loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=my_collate)

  # optimizer
  optimizer = optim.Adam(model.parameters(), lr=cfg.TRAINING.LR, weight_decay=cfg.TRAINING.WT_DECAY) # todo tune WD
  print("number of params: ", sum(p.numel() for p in model.parameters()))
  
  # Loss
  criterion = CustomLoss(cfg)

  model.train()
  for k in range(cfg.TRAINING.ITER):
    for b, (x, target, cls_indices) in enumerate(loader, 0):
      log_p_sum, logdet, z, mean, log_sd = model(x)
      
      nll_loss = criterion(logdet, log_p_sum)
      c_loss = criterion.closs(z, target, cls_indices)

      loss = nll_loss 

      model.zero_grad()
      loss.backward()
      optimizer.step()

      if k % 100 == 0:
        print(nll_loss.item(), c_loss.item(), loss.item())
        z_rec = torch.normal(mean, log_sd.exp())
        ut.plot(z.detach(), target, k)
        ut.plot_3d(z.detach(), log_p_sum.detach(), k, target)
        ut.plot(model.reverse(z_rec).detach(), target, f"recon_{k}.png")



