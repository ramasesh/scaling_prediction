#!/usr/bin/env python

import pathlib
import time
import json
from absl import logging, app, flags
logging.set_verbosity(logging.INFO)

import numpy as np
import random

import torch
import torch.nn as nn
import torchvision

from src.dataloader import get_loader
from src import utils, reporters
from src.utils import str2bool, str2splittasklist, AverageMeter
from src.argparser import parse_args
from src import reporters
from src.models import multitaskmodels as mtm
from src.train_utils import train, test, update_state
from src import train_setup
from src.model_measurements import utils as mu

import copy

FLAGS = flags.FLAGS
flags.DEFINE_boolean('debug', False, 'Produces debugging output')

def make_measurements(model, config, loaders, initial_parameters, initial_named_parameters):
  train_loader, test_loader, single_train_loader, single_test_loader = loaders
  measurements = {}

  device = torch.device(config['run_config']['device'])
  grad_characteristics_to_measure = ['L2Norm',
                                     'L1Norm',
                                     'ApproxSpectralNorm',
                                     'LinftyNorm']
  """ Data-independent """
  """ Model characteristics"""
  characteristics_to_measure = ['L2Norm',
                                'L1Norm',
                                'ApproxSpectralNorm',
                                'LinftyNorm',
                                'L2toInit']
  print('Measuring model characteristics')
  model_characteristics = mu.measure_model_characteristics(characteristics_to_measure,
                                                           model,
                                                           model.params_to_measure,
                                                           initial_named_parameters)
  measurements.update(model_characteristics)
  """ End model characteristics """

  """ Data-dependent """
  """ Output characteristics only """
  outputs_to_measure = ['logit_sum',
                        'correct_logit',
                        'logit_margin',
                        'highest_incorrect_logit',
                        'accuracy',
                        'cross_entropy']
  print('Measuring output characteristics')
  for ind, l in enumerate([train_loader, test_loader]):
      output_cumulants = mu.measure_on_dataset(outputs_to_measure,
                                                              model,
                                                              l,
                                                              device,
                                                              initial_parameters)
      if ind == 0:
        measurements.update({'train_outputs': output_cumulants})
      elif ind == 1:
        measurements.update({'test_outputs': output_cumulants})
  """ End output characteristics """

  """ Data-dependent """
  """ Norms of gradients """
  for ind, l in enumerate([single_train_loader, single_test_loader]):
      print('Measuring gradient characteristics')
      ## debug
      internal_characteristics = mu.measure_characteristic_on_dataset(
                                            grad_characteristics_to_measure,
                                            model,
                                            l,
                                            device,
                                            model.params_to_measure,
                                            list(initial_named_parameters))
      if ind == 0:
        measurements.update({'train_gradient_chars': internal_characteristics})
      elif ind == 1:
        measurements.update({'test_gradient_chars': internal_characteristics})
  return measurements

def main(argv):
  config = parse_args()
  logging.info(json.dumps(config, indent=2))

  if FLAGS.debug:
    print('non-flag arguments:', argv)
    return

  reporters.save_config(config)

  # set up reporting
  data_store = {}
  reporter = reporters.build_reporters(config['save_config'],
                                       data_store)
  prefixes = ['test', 'train', 'model_measurements']
  reporter = reporters.prefix_reporters(reporter, prefixes)

  loaders = get_loader(config['data_config'])
  train_loader, test_loader, single_train_loader, single_test_loader = loaders
  config['optim_config']['steps_per_epoch'] = len(train_loader)

  train_setup.set_reproducible(seed=config['run_config']['seed'])

  logging.info('Loading model...')
  model = utils.load_model(config['model_config'])
  initial_parameters = copy.deepcopy(list(model.parameters()))
  initial_named_parameters = copy.deepcopy(list(model.named_parameters()))
  device = torch.device(config['run_config']['device'])
  if device.type == 'cuda' and torch.cuda.device_count() > 1:
      model = nn.DataParallel(model)
  model.to(device)
  logging.info('Done')

  train_criterion = nn.CrossEntropyLoss(reduction='mean')
  test_criterion = nn.CrossEntropyLoss(reduction='mean')

  # run test before start training
  epoch = -1
  logging.info('Initial evaluation')
  test_log = test(model, test_criterion, test_loader, config['run_config'])
  reporter['test'].report_all(epoch, test_log)

  model_dicts = {}

  optimizer, scheduler = utils.create_optimizer(model.parameters(),
                                                config['optim_config'])

  logging.info('Beginning training')

  for epoch in range(config['optim_config']['epochs']):
    train_log = train(model, optimizer, scheduler, train_criterion,
                      train_loader, config['run_config'])
    reporter[f'train'].report_all(epoch, train_log)

    test_log = test(model, test_criterion, test_loader, config['run_config'])
    reporter[f'test'].report_all(epoch, test_log)

    if should_measure(epoch, config):
        model_measurements = make_measurements(model, config, loaders,
                                               initial_parameters, initial_named_parameters)

        reporter['model_measurements'].report_all(epoch, model_measurements)

  reporters.save_dict(config, model_dicts, 'model_parameters')

def should_measure(epoch, config):
  return epoch % config['save_config']['measure_every'] == 0

if __name__ == '__main__':
  app.run(main)

