import torch
import torch.nn as nn
import numpy as np

def L1_norm(mat1, **kwargs):
  return mat1.norm(p=1).item()

def Linfty_norm(mat1, **kwargs):
  return mat1.norm(p=float('inf')).item()

def approx_spectral_norm(mat1, **kwargs):
  if len(mat1.size()) == 4:
    reshaped_mat = mat1.view(-1,mat1.size()[-1]).numpy()
    return np.max(np.linalg.svd(reshaped_mat, compute_uv=False))
  elif len(mat1.size()) == 2:
    _, S, _ = mat1.svd()
    return torch.max(S).item()
  elif len(mat1.size()) == 1:
    return mat1.norm(p=2).item()

def spectral_norm(mat1, **kwargs):
  if len(mat1.size()) == 4:
    # using results in https://arxiv.org/pdf/1805.10408.pdf
    input_shape = kwargs['input_shape']
    transform = np.fft.fft2(mat1.cpu().numpy(), input_shape, axes=[0,1])
    return np.max(np.linalg.svd(transform, compute_uv=False))
  elif len(mat1.size()) == 2:
    _, S, _ = mat1.svd()
    return torch.max(S).item()
  elif len(mat1.size()) == 1:
    return mat1.norm(p=2).item()

def L2_norm(mat1, **kwargs):
  return mat1.norm(p=2).item()

def Euclidean_distance(mat1, mat2, **kwargs):
  return L2_norm(mat1.cpu() - mat2.cpu())

