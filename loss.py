import torch 
from torch import nn 

from icecream import ic
from sys import exit as e


def off_diagonal(x):
  # return a flattened view of the off-diagonal elements of a square matrix
  n, m = x.shape
  assert n == m
  return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()



class CustomLoss(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg

  def forward(self, logdet, log_p_sum):
    logprob = logdet + log_p_sum
    loss = -torch.sum(logprob) # NLL
    return loss

  def btwins(self, z, z2 = None, invariance_flag=True):

    if invariance_flag:
      c_a = z.T @ z
      c_a.div_(self.cfg.TRAINING.BATCH)
      on_diag = torch.diagonal(c_a).add_(-1).pow_(2).sum()
      off_diag = off_diagonal(c_a).pow_(2).sum()
      loss = on_diag + self.cfg.LOSS.LAMBDA * off_diag
    else:
      c_a = z.T @ z2
      c_a.div_(self.cfg.TRAINING.BATCH)
      on_diag = torch.diagonal(c_a).pow_(2).sum()
      loss = on_diag

    return loss

  def closs(self, z, target, cls_indices):
    mask1 = torch.zeros_like(z, requires_grad=False)
    mask1[cls_indices[0]] = 1

    mask2 = torch.zeros_like(z, requires_grad=False)
    mask2[cls_indices[1]] = 1

    z1 = z * mask1
    z2 = z * mask2

    z1 = (z1 - z1.mean(0)) / z1.std(0) # NxD
    z2 = (z2 - z2.mean(0)) / z2.std(0) # NxD
    
    c_a = self.btwins(z1)
    c_b = self.btwins(z2)
    c_ab = self.btwins(z1, z2=z2, invariance_flag=False)

    loss = c_a + c_b
    return loss

