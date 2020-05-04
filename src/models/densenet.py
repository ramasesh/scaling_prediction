'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from absl import app
from absl import flags

arguments = ['growth_rate_dn', 'n_blocks_dn', 'compression_rate_dn']
json_arguments = ['n_blocks_dn']

flags.DEFINE_integer('growth_rate_dn', 6, 'DenseNet growth rate.')
flags.DEFINE_string('n_blocks_dn', "[6,12,24,16]", 'DenseNet num blocks.' )
flags.DEFINE_float('compression_rate_dn', 0.5, 'DenseNet compression rate.')

def initialize_weights(m):
  if isinstance(m, nn.Conv2d):
    nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
  elif isinstance(m, nn.BatchNorm2d):
    m.weight.data.fill_(1)
    m.bias.data.zero_()
  elif isinstance(m, nn.Linear):
    m.bias.data.zero()

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x), inplace=True))
        out = F.avg_pool2d(out, 2)
        return out


class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()

        input_shape = config['input_shape']
        n_classes = config['n_classes']
        n_blocks = config['n_blocks_dn']
        self.growth_rate = config['growth_rate_dn']
        compression_rate = config['compression_rate_dn']

        num_planes = 2*self.growth_rate
        self.conv0 = nn.Conv2d(input_shape[1], num_planes, kernel_size=3, padding=1, bias=False)

        self.dense0 = self._make_dense_layers(Bottleneck, num_planes, n_blocks[0])
        num_planes += n_blocks[0]*self.growth_rate
        out_planes = int(math.floor(num_planes*compression_rate))
        self.trans0 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense1 = self._make_dense_layers(Bottleneck, num_planes, n_blocks[1])
        num_planes += n_blocks[1]*self.growth_rate
        out_planes = int(math.floor(num_planes*compression_rate))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(Bottleneck, num_planes, n_blocks[2])
        num_planes += n_blocks[2]*self.growth_rate
        out_planes = int(math.floor(num_planes*compression_rate))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(Bottleneck, num_planes, n_blocks[3])
        num_planes += n_blocks[3]*self.growth_rate

        self.bn = nn.BatchNorm2d(num_planes)

        self.fc = nn.Linear(num_planes, n_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv0(x)
        out = self.trans0(self.dense0(out))
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = F.avg_pool2d(F.relu(self.bn(out), inplace=True), 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def DenseNet121():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def DenseNet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

def densenet_cifar():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12)

def test():
    net = densenet_cifar()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y)

# test()
