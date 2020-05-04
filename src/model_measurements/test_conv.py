import torch
import torch.nn as nn
import numpy as np

import test_utils as t
import utils as u

input_width = 10
input_channels = 2
n_outputs = 5
dataset_size = 50
batch_size = 5

conv_model = t.TestConv()

modules_to_measure = {'conv.weight': {'input_shape': [input_width, input_width]},
                      'conv.bias': {'input_shape': [input_width, input_width]},
                      'fc.weight': {},
                      'fc.bias': {}}

test_data = torch.randn([dataset_size, input_channels, input_width, input_width])
test_labels = torch.randint(low=0,
                            high=n_outputs,
                            size=[dataset_size])

test_dataset = list(zip(test_data, test_labels))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
grad_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

outputs_to_measure = ['logit_sum',
                      'correct_logit',
                      'logit_margin',
                      'highest_incorrect_logit',
                      'accuracy',
                      'cross_entropy']

characteristics_to_measure = ['L2Norm',
                              'L1Norm',
                              'SpectralNorm',
                              'LinftyNorm',
                              'L2toInit']

measured_cumulants = u.measure_on_dataset(outputs_to_measure,
                                           conv_model,
                                           test_loader,
                                           torch.device('cpu'),
                                           conv_model.parameters())

measured_characteristics = u.measure_model_characteristics(characteristics_to_measure,
                                                            conv_model,
                                                            modules_to_measure,
                                                            conv_model.named_parameters())


measured_characteristics_internal = u.measure_characteristic_on_dataset(characteristics_to_measure,
                                                 conv_model,
                                                 grad_test_loader,
                                                 torch.device('cpu'),
                                                 modules_to_measure,
                                                 list(conv_model.named_parameters()))
