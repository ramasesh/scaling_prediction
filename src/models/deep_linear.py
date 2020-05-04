import torch
import torch.nn as nn
import numpy as np

from absl import app
from absl import flags

arguments = ['width_dl', 'depth_dl', 'use_bias_dl']
json_arguments = []

flags.DEFINE_integer('width_dl', 20, 'Deep linear model width.')
flags.DEFINE_integer('depth_dl', 10, 'Depth of resnet or deep linear model.')
flags.DEFINE_boolean('use_bias_dl', True, 'Use bias in deep linear model?')

def initialize_weights(module):
  if isinstance(module, nn.Linear):
    nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='linear')
    module.bias.data.zero_()

class Network(nn.Module):
  def __init__(self, config):
    super(Network, self).__init__()

    self.input_shape = config['input_shape']
    self.n_classes = config['n_classes']

    self.width = config['width_dl']
    self.depth = config['depth_dl']
    self.bias = config['use_bias_dl']

    self.__make_stages__()

  def __count_input_features__(self):
    assert self.input_shape[0] == 1
    return np.prod(self.input_shape)

  def __make_stages__(self):
    stages = {}

    in_features = self.__count_input_features__()

    initial_stage = nn.Linear(in_features = in_features,
                              out_features = self.width,
                              bias = self.bias)
    stages[0] = initial_stage

    for l in range(1, self.depth-1):
      stage = nn.Linear(in_features = self.width,
                        out_features = self.width,
                        bias = self.bias)
      stages[l] = stage

    stages = {str(k):v for k,v in stages.items()}
    # ModuleDict() requires keys to be strings

    self.stages = nn.ModuleDict(stages)

  def forward(self, x):
    x = x.view(x.size(0), -1)
    for i in range(self.depth-1):
      x = self.stages[str(i)](x)

    return x
