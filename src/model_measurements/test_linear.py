import torch
import torch.nn as nn
import numpy as np

import utils as u

input_dim = 10
n_hidden = 20
n_outputs = 2
dataset_size = 50
batch_size = 5
LinearTest = nn.Sequential(nn.Linear(input_dim, n_hidden),
                           nn.Linear(n_hidden, n_outputs))

modules_to_study = {'0.weight': {},
                    '0.bias': {},
                    '1.weight': {},
                    '1.bias': {}}

test_data = torch.randn([dataset_size, input_dim])
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
                                           LinearTest,
                                           test_loader,
                                           torch.device('cpu'),
                                           LinearTest.parameters())

measured_characteristics = u.measure_model_characteristics(characteristics_to_measure,
                                                            LinearTest,
                                                            modules_to_study,
                                                            LinearTest.named_parameters())


measured_characteristics_internal = u.measure_characteristic_on_dataset(characteristics_to_measure,
                                                 LinearTest,
                                                 grad_test_loader,
                                                 torch.device('cpu'),
                                                 modules_to_study,
                                                 list(LinearTest.named_parameters()))
