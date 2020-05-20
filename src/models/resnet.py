import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools


from absl import app
from absl import flags

arguments = ['base_channels_rn', 'block_type_rn', 'depth_rn']
json_arguments = []

flags.DEFINE_integer('base_channels_rn', 32, 'ResNet base channels.')
flags.DEFINE_string('block_type_rn', 'basic', 'ResNet block type.')
flags.DEFINE_integer('depth_rn', 18, 'ResNet depth.')

def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,  # downsample with first conv
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))  # BN

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)  # apply ReLU after addition
        return y


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride):
        super(BottleneckBlock, self).__init__()

        bottleneck_channels = out_channels // self.expansion

        self.conv1 = nn.Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,  # downsample with 3x3 conv
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        self.conv3 = nn.Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()  # identity
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))  # BN

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = F.relu(self.bn2(self.conv2(y)), inplace=True)
        y = self.bn3(self.conv3(y))  # not apply ReLU
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)  # apply ReLU after addition
        return y


class Network(nn.Module):

    params_to_measure = {'conv.weight': {},
                         'bn.weight': {},
                         'bn.bias': {},
                         'stage0.block1.conv1.weight': {},
                         'stage0.block1.bn1.weight': {},
                         'stage0.block1.bn1.bias': {},
                         'stage0.block1.conv2.weight': {},
                         'stage0.block1.bn2.weight': {},
                         'stage0.block1.bn2.bias': {},
                         'stage0.block2.conv1.weight': {},
                         'stage0.block2.bn1.weight': {},
                         'stage0.block2.bn1.bias': {},
                         'stage0.block2.conv2.weight': {},
                         'stage0.block2.bn2.weight': {},
                         'stage0.block2.bn2.bias': {},
                         'stage1.block1.conv1.weight': {},
                         'stage1.block1.bn1.weight': {},
                         'stage1.block1.bn1.bias': {},
                         'stage1.block1.conv2.weight': {},
                         'stage1.block1.bn2.weight': {},
                         'stage1.block1.bn2.bias': {},
                         'stage1.block1.shortcut.conv.weight': {},
                         'stage1.block1.shortcut.bn.weight': {},
                         'stage1.block1.shortcut.bn.bias': {},
                         'stage1.block2.conv1.weight': {},
                         'stage1.block2.bn1.weight': {},
                         'stage1.block2.bn1.bias': {},
                         'stage1.block2.conv2.weight': {},
                         'stage1.block2.bn2.weight': {},
                         'stage1.block2.bn2.bias': {},
                         'stage2.block1.conv1.weight': {},
                         'stage2.block1.bn1.weight': {},
                         'stage2.block1.bn1.bias': {},
                         'stage2.block1.conv2.weight': {},
                         'stage2.block1.bn2.weight': {},
                         'stage2.block1.bn2.bias': {},
                         'stage2.block1.shortcut.conv.weight': {},
                         'stage2.block1.shortcut.bn.weight': {},
                         'stage2.block1.shortcut.bn.bias': {},
                         'stage2.block2.conv1.weight': {},
                         'stage2.block2.bn1.weight': {},
                         'stage2.block2.bn1.bias': {},
                         'stage2.block2.conv2.weight': {},
                         'stage2.block2.bn2.weight': {},
                         'stage2.block2.bn2.bias': {},
                         'stage3.block1.conv1.weight': {},
                         'stage3.block1.bn1.weight': {},
                         'stage3.block1.bn1.bias': {},
                         'stage3.block1.conv2.weight': {},
                         'stage3.block1.bn2.weight': {},
                         'stage3.block1.bn2.bias': {},
                         'stage3.block1.shortcut.conv.weight': {},
                         'stage3.block1.shortcut.bn.weight': {},
                         'stage3.block1.shortcut.bn.bias': {},
                         'stage3.block2.conv1.weight': {},
                         'stage3.block2.bn1.weight': {},
                         'stage3.block2.bn1.bias': {},
                         'stage3.block2.conv2.weight': {},
                         'stage3.block2.bn2.weight': {},
                         'stage3.block2.bn2.bias': {},
                         'fc.weight': {},
                         'fc.bias': {}}

    def __init__(self, config):
        super(Network, self).__init__()

        input_shape = config['input_shape']
        n_classes = config['n_classes']

        base_channels = config['base_channels_rn']
        block_type = config['block_type_rn']
        depth = config['depth_rn']

        assert block_type in ['basic', 'bottleneck']
        if block_type == 'basic':
            block = BasicBlock
            n_blocks_per_stage = (depth - 2) // 8
            assert n_blocks_per_stage * 8 + 2 == depth
        else:
            block = BottleneckBlock
            n_blocks_per_stage = (depth - 2) // 12
            assert n_blocks_per_stage * 12 + 2 == depth

        n_channels = [
            base_channels,
            base_channels * 2 * block.expansion,
            base_channels * 4 * block.expansion,
            base_channels * 8 * block.expansion
        ]

        self.conv = nn.Conv2d(
            input_shape[1],
            n_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn = nn.BatchNorm2d(base_channels)

        self.stage0 = self._make_stage(
            n_channels[0], n_channels[0], n_blocks_per_stage, block, stride=1)
        self.stage1 = self._make_stage(
            n_channels[0], n_channels[1], n_blocks_per_stage, block, stride=2)
        self.stage2 = self._make_stage(
            n_channels[1], n_channels[2], n_blocks_per_stage, block, stride=2)
        self.stage3 = self._make_stage(
            n_channels[2], n_channels[3], n_blocks_per_stage, block, stride=2)

        with torch.no_grad():
            self.feature_size = self._forward_conv(torch.zeros(*input_shape)).view(-1).shape[0]

        self.fc = nn.Linear(self.feature_size, n_classes)

        # initialize weights
        self.apply(initialize_weights)

    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = 'block{}'.format(index + 1)
            if index == 0:
                stage.add_module(block_name,
                                 block(in_channels,
                                       out_channels,
                                       stride=stride))
            else:
                stage.add_module(block_name,
                                 block(out_channels,
                                       out_channels,
                                       stride=1))
        return stage

    def _forward_conv(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
