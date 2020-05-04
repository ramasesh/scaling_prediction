# Lint as: python3
"""TODO(ramasesh): DO NOT SUBMIT without one-line documentation for testargparser.

TODO(ramasesh): DO NOT SUBMIT without a detailed description of testargparser.
"""

from src import utils

import importlib
import json
import torch
from collections import OrderedDict

from absl import app
from absl import flags

allowed_architectures = ['deep_linear', 'densenet', 'resnet', 'vgg']
for arch in allowed_architectures:
  importlib.import_module('src.models.{}'.format(arch))

# model config
flags.DEFINE_enum('arch', 'vgg', allowed_architectures, 'Model architecture.')
flags.mark_flag_as_required('arch')
# data config
flags.DEFINE_boolean('use_horizontal_flip', False, 'Use horizontal flip')
flags.DEFINE_boolean('use_random_crop', False, 'Use random crop')
flags.DEFINE_integer('aug_crop_padding', 4, 'Aug crop padding')
flags.DEFINE_float('aug_flip_probability', 0.5, 'Aug flip probability')
flags.DEFINE_enum('dataset', 'CIFAR10', ['CIFAR10', 'CIFAR100', 'MNIST'], 'Dataset')
flags.DEFINE_integer('num_workers', 7, 'Number of dataloader workers')
flags.DEFINE_integer('examples_per_class', None, 'Examples per class')
flags.DEFINE_integer('subsampling_seed', 0, 'Seed for selecting random data sample')
# optim config
flags.DEFINE_enum('optimizer', 'sgd', ['sgd'], 'Optimizer')
flags.DEFINE_enum('scheduler', 'None', ['None', 'cosine', 'multistep'], 'Scheduler')
flags.DEFINE_float('lr_decay', 0.1, 'Decay rate for multistep scheduler')
flags.DEFINE_integer('epochs', 30, 'Number of epochs')
flags.DEFINE_string('milestones', '[10]', 'Decay milestones for multistep scheduler')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_float('base_lr', 0.01, 'Learning rate')
flags.DEFINE_float('weight_decay', 0.0001, 'Weight decay')
flags.DEFINE_float('momentum', 0.9, 'Momentum parameter')
flags.DEFINE_boolean('nesterov', True, 'Use nesterov acceleration')
# run config
flags.DEFINE_integer('seed', 0, 'Seed')
flags.DEFINE_enum('device', 'cuda',  ['cuda','cpu'], 'Device')
# save config
flags.DEFINE_string('save_location', '.', 'Save location')
flags.DEFINE_string('job_name_template', 'scaling_arch-{arch}', 'Job name template')
flags.DEFINE_integer('measure_every', 10, 'Measure model params every how-many epochs')

FLAGS = flags.FLAGS

input_shapes = {'CIFAR10': (1,3,32,32),
                'CIFAR100': (1,3,32,32),
                'MNIST': (1,1,28,28)}

n_classes = {'CIFAR10': 10,
             'CIFAR100': 100,
             'MNIST': 10}

def parse_args():
  config = {}
  config['model_config'] = get_model_config()
  config['data_config'] = get_data_config()
  config['env_info'] = _get_env_info()
  config['optim_config'] = get_optim_config()
  config['run_config'] = get_run_config()
  config['save_config'] = get_save_config()

  config = populate_save_location(config)

  return config

def get_model_config():
  model_config = {}
  model_config['arch'] = FLAGS.arch

  module = importlib.import_module('src.models.{}'.format(model_config['arch']))
  model_arguments = getattr(module, 'arguments')
  model_json_arguments = getattr(module, 'json_arguments')

  for argument in model_arguments:
    model_config[argument] = getattr(FLAGS, argument)
    if argument in model_json_arguments and isinstance(model_config[argument],str):
      model_config[argument] = json.loads(model_config[argument])

  # Pull from task-specific flags
  model_config['input_shape'] = input_shapes[FLAGS.dataset]
  model_config['n_classes'] = n_classes[FLAGS.dataset]

  return model_config


def get_data_config():
  data_config = {}
  data_args = ['use_horizontal_flip', 'use_random_crop', 'aug_crop_padding', 'aug_flip_probability',
               'dataset', 'num_workers', 'batch_size', 'examples_per_class', 'subsampling_seed']

  for arg in data_args:
    data_config[arg] = getattr(FLAGS, arg)

  data_config['use_gpu'] = FLAGS.device != 'cpu'

  return data_config

def _get_env_info():
  info = OrderedDict({
    'pytorch_version': torch.__version__,
    'cuda_version': torch.version.cuda,
    'cudnn_version': torch.backends.cudnn.version(),
  })

  def _get_device_info(device_id):
    name = torch.cuda.get_device_name(device_id)
    capability = torch.cuda.get_device_capability(device_id)
    capability = '{}.{}'.format(*capability)
    return name, capability

  if FLAGS.device != 'cpu':
    for gpu_id in range(torch.cuda.device_count()):
      name, capability = _get_device_info(gpu_id)
      info['gpu{}'.format(gpu_id)] = OrderedDict({
        'name':
        name,
        'capability':
        capability,
      })

  return info

def get_optim_config():
  optim_args = ['epochs', 'batch_size', 'optimizer', 'base_lr', 'weight_decay',
                'momentum', 'nesterov', 'scheduler', 'lr_decay', 'milestones']
  optim_config = {}
  for arg in optim_args:
    optim_config[arg] = getattr(FLAGS, arg)

  optim_config['milestones'] = json.loads(optim_config['milestones'])
  return optim_config

def get_run_config():
  run_args = ['seed', 'device']
  run_config = {}
  for arg in run_args:
    run_config[arg] = getattr(FLAGS, arg)

  return run_config

def get_save_config():
  save_args = ['save_location', 'job_name_template', 'measure_every']
  save_config = {}
  for arg in save_args:
    save_config[arg] = getattr(FLAGS, arg)
  return save_config


def populate_save_location(config):
    all_arguments = {}
    for key in config.keys():
        all_arguments.update(config[key])

    config['save_config'].update({'job_name': config['save_config']['job_name_template'].format(**all_arguments)})

    return config


