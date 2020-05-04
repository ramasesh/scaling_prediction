import pathlib
import numpy as np

import torch
import torch.utils.data

import torchvision
import torchvision.models
import torchvision.transforms

from src import transforms

import json

import copy

class Dataset:
    torchvision_datasets = ['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST', 'KMNIST']
    def __init__(self, config):
        self.config = config
        dataset_rootdir = pathlib.Path('./data')
        self.dataset_dir = dataset_rootdir / config['dataset']

        self._train_transforms = []
        self.train_transform = self._get_train_transform()
        self.test_transform = self._get_test_transform()

    def get_datasets(self):
        if self.config['dataset'] in self.torchvision_datasets:
            train_dataset = getattr(torchvision.datasets, self.config['dataset'])(
                            self.dataset_dir,
                            train=True,
                            transform=self.train_transform,
                            download=True)
            test_dataset = getattr(torchvision.datasets, self.config['dataset'])(
                            self.dataset_dir,
                            train=False,
                            transform=self.test_transform,
                            download=True)
        else:
            raise Exception('Dataset not supported.')
        return train_dataset, test_dataset

    def _add_random_crop(self):
        transform = torchvision.transforms.RandomCrop(
            self.size, padding=self.config['aug_crop_padding'])
        self._train_transforms.append(transform)

    def _add_horizontal_flip(self):
        transform = torchvision.transforms.RandomHorizontalFlip(p=self.config['aug_flip_probability'])
        self._train_transforms.append(transform)

    def _add_normalization(self):
        self._train_transforms.append(
            transforms.Normalize(self.mean, self.std))

    def _add_to_tensor(self):
        self._train_transforms.append(transforms.ToTensor())

    def _get_train_transform(self):
        if self.config.get('use_random_crop', False):
            self._add_random_crop()
        if self.config.get('use_horizontal_flip', False):
            self._add_horizontal_flip()
        self._add_normalization()
        self._add_to_tensor()
        return torchvision.transforms.Compose(self._train_transforms)

    def _get_test_transform(self):
        transform = torchvision.transforms.Compose([
            transforms.Normalize(self.mean, self.std),
            transforms.ToTensor(),
        ])
        return transform

class CIFAR(Dataset):
  def __init__(self, config):
    self.size = 32
    if 'CIFAR100' in config['dataset']:
      self.mean = np.array([0.5071, 0.4865, 0.4409])
      self.std = np.array([0.2673, 0.2564, 0.2762])
    elif 'CIFAR10' in config['dataset']:
      self.mean = np.array([0.4914, 0.4822, 0.4465])
      self.std = np.array([0.2470, 0.2435, 0.2616])
    super(CIFAR, self).__init__(config)


class MNIST(Dataset):
  def __init__(self, config):
    self.size = 28
    if 'FashionMNIST' in config['dataset']:
      self.mean = np.array([0.2860])
      self.std = np.array([0.3530])
    elif 'KMNIST' in config['dataset']:
      self.mean = np.array([0.1904])
      self.std = np.array([0.3475])
    elif 'MNIST' in config['dataset']:
      self.mean = np.array([0.1307])
      self.std = np.array([0.3081])
    super(MNIST, self).__init__(config)


class DownsampledDataset():
  # Dataset with a subset of images from the original labeled dataset,
  # with an equal number of images per class

  def __init__(self, full_dataset, num_pts_per_class, label_locations, random_seed=0):
    self.full_dataset = full_dataset
    self.label_locations = label_locations

    self.__set_selected_indices__(random_seed, num_pts_per_class)

  def __set_selected_indices__(self, random_seed, num_pts_per_class):
    np.random.seed(random_seed)
    self.selected_indices = []
    for key in self.label_locations.keys():
      self.selected_indices.append(np.random.choice(self.label_locations[key],
                                                    size=num_pts_per_class,
                                                    replace=False))
    self.selected_indices = np.array(self.selected_indices, dtype=int)
    self.selected_indices = self.selected_indices.flatten('F') # column-major flattening

  def __len__(self):
    return len(self.selected_indices)

  def __getitem__(self, idx):
    return self.full_dataset.__getitem__(self.selected_indices[idx])

def worker_init_fn(worker_id):
  np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_loader(config):
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    use_gpu = config['use_gpu']

    dataset_name = config['dataset']

    if 'CIFAR' in dataset_name:
        dataset = CIFAR(config)
    elif 'MNIST' in dataset_name:
        dataset = MNIST(config)

    train_dataset, test_dataset = dataset.get_datasets()

    if config['examples_per_class'] is not None:
        print('Downsampling')
        examples_per_class = config['examples_per_class']

        with open('src/dataset_indices.json') as f:
            label_locations = json.load(f)
        label_locations = label_locations[dataset_name]['train']

        train_dataset = DownsampledDataset(train_dataset, examples_per_class,
                                           label_locations, random_seed = config['subsampling_seed'])


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=use_gpu,
        drop_last=False,
    )
    single_train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    single_test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=use_gpu,
        drop_last=False,
    )
    return train_loader, test_loader, single_train_loader, single_test_loader

