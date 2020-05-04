# Lint as: python3
"""TODO(ramasesh): DO NOT SUBMIT without one-line documentation for train_setup.

TODO(ramasesh): DO NOT SUBMIT without a detailed description of train_setup.
"""
import torch
import random, torch, os, numpy as np

def set_reproducible(seed):
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  set_seed(seed)

def set_seed(seed):
  # seed everything
  # not sure if this is all needed, or even sufficient
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)

