import torch
import torch.nn as nn
import numpy as np

import utils as u

class TestConv(nn.Module):

  def __init__(self):
    super(TestConv, self).__init__()

    n_input_channels = 2
    n_output_channels = 4
    input_width = 10
    input_shape = (1, n_input_channels, input_width, input_width)
    n_classes = 5

    self.conv = nn.Conv2d(n_input_channels,
                          n_output_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=True)

    with torch.no_grad():
      self.feature_size = self.conv(torch.zeros(*input_shape)).view(-1).shape[0]

    self.fc = nn.Linear(self.feature_size, n_classes)

  def forward(self, x):
    x = self.conv(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x

