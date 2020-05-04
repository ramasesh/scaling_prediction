import torch
import torch.nn as nn
from src.model_measurements import norms

def measure_logit_sum(model_outputs, model_parameters, targets, initial_parameters, name=None, msmt_type=None, **kwargs):
  return torch.sum(model_outputs, dim=1)

def measure_correct_logit(model_outputs, model_parameters, targets, initial_parameters, name=None, msmt_type=None, **kwargs):
  return model_outputs[torch.arange(model_outputs.size(0)), targets]

def measure_logit_margin(model_outputs, model_parameters, targets, initial_parameters, name=None, msmt_type=None, **kwargs):
  return measure_correct_logit(model_outputs, model_parameters, targets, initial_parameters) - measure_highest_incorrect_logit(model_outputs, model_parameters, targets, initial_parameters)

def measure_highest_incorrect_logit(model_outputs, model_parameters, targets, initial_parameters, name=None, msmt_type=None, **kwargs):
  " sets correct logit to -infinity and takes the highest logit "
  negative_inf = -1*float('inf')
  cloned_outputs = model_outputs.clone()
  for i in range(cloned_outputs.size(0)):
    cloned_outputs[i, targets[i]] = negative_inf

  maxes, _ = torch.max(cloned_outputs, dim=1)
  return maxes

def measure_accuracy(model_outputs, model_parameters, targets, initial_parameters, name=None, msmt_type=None, **kwargs):
  _, preds = torch.max(model_outputs, dim=1)
  return preds.eq(targets).float()

def measure_cross_entropy(model_outputs, model_parameters, targets, initial_parameters, name=None, msmt_type=None, **kwargs):
  loss = nn.CrossEntropyLoss(reduction='none')
  return loss(model_outputs, targets)

### Measurement functions ###
def measure_L2_norm(model_outputs, current_parameters, targets, initial_parameters, name, msmt_type='param', **kwargs):
  value = extract_named_parameter(current_parameters, name, msmt_type)
  fun = norms.L2_norm
  return apply_correctly(fun, msmt_type, value, **kwargs)

def measure_L1_norm(model_outputs, current_parameters, targets, initial_parameters, name, msmt_type='param', **kwargs):
  value = extract_named_parameter(current_parameters, name, msmt_type)
  fun = norms.L1_norm
  return apply_correctly(fun, msmt_type, value, **kwargs)

def measure_spectral_norm(model_outputs, current_parameters, targets, initial_parameters, name, msmt_type='param', **kwargs):
  value = extract_named_parameter(current_parameters, name, msmt_type)
  fun = norms.spectral_norm
  return apply_correctly(fun, msmt_type, value, **kwargs)

def measure_approx_spectral_norm(model_outputs, current_parameters, targets, initial_parameters, name, msmt_type='param', **kwargs):
  value = extract_named_parameter(current_parameters, name, msmt_type)
  fun = norms.approx_spectral_norm
  return apply_correctly(fun, msmt_type, value, **kwargs)

def measure_Linfty_norm(model_outputs, current_parameters, targets, initial_parameters, name, msmt_type='param', **kwargs):
  value = extract_named_parameter(current_parameters, name, msmt_type)
  fun = norms.Linfty_norm
  return apply_correctly(fun, msmt_type, value, **kwargs)

def measure_L2toInit(model_outputs, current_parameters, targets, initial_parameters, name, msmt_type='param', **kwargs):
  current_value = extract_named_parameter(current_parameters, name, msmt_type)
  init_value = extract_named_parameter(initial_parameters, name, msmt_type)
  fun = norms.Euclidean_distance
  return apply_correctly_2arg(fun, msmt_type, current_value, init_value, **kwargs)

def extract_named_parameter(parameters, name, msmt_type):
  for p in parameters:
    if p[0] == name:
      param = p[1]
      break
  else:
    raise Exception('Did not find a match')
  if msmt_type == 'grad':
    return p[1].grad.data
  elif msmt_type == 'param':
    return p[1].data

def apply_correctly(fun, msmt_type, value, **kwargs):
  if msmt_type == 'param':
    return fun(value, **kwargs)
  elif msmt_type == 'grad':
    return torch.Tensor([fun(value, **kwargs)])

def apply_correctly_2arg(fun, msmt_type, value1, value2, **kwargs):
  if msmt_type == 'param':
    return fun(value1, value2, **kwargs)
  elif msmt_type == 'grad':
    return torch.Tensor([fun(value1, value2, **kwargs)])

possible_measurements = {'logit_sum': measure_logit_sum,
                         'correct_logit': measure_correct_logit,
                         'logit_margin': measure_logit_margin,
                         'highest_incorrect_logit': measure_highest_incorrect_logit,
                         'accuracy': measure_accuracy,
                         'cross_entropy': measure_cross_entropy}

