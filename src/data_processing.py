# Lint as: python3
"""Functions for processing (i.e., subsampling etc) datasets

TODO(ramasesh): DO NOT SUBMIT without a detailed description of data_processing.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision
import numpy as onp

SUPPORTED_DATASETS={'CIFAR10': torchvision.datasets.CIFAR10,
                    'CIFAR100': torchvision.datasets.CIFAR100,
                    'MNIST': torchvision.datasets.MNIST}

def get_dataset_labels(dataset_spec,
                       train=True):
  """ Grabs the labels of the items in the requested dataset and returns them"""

  dataset_object = SUPPORTED_DATASETS[dataset_spec](root='./data',
                                                    train=train,
                                                    download=True)

  n_classes = len(dataset_object.classes)
  labels_by_image = dataset_object.targets

  images_by_class = onp.array([onp.where(labels_by_image==i)[0] for i in range(n_classes)])

  return images_by_class


