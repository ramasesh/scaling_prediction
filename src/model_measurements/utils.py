import torch
import torch.nn as nn
import collections
from src.model_measurements.meters import MomentAggregator, get_cumulants
from src.model_measurements import measurements as om

measurement_functions = {'L2Norm': om.measure_L2_norm,
                         'L1Norm': om.measure_L1_norm,
                         'SpectralNorm': om.measure_spectral_norm,
                         'ApproxSpectralNorm': om.measure_approx_spectral_norm,
                         'LinftyNorm': om.measure_Linfty_norm,
                         'L2toInit': om.measure_L2toInit,
                         'logit_sum': om.measure_logit_sum,
                         'correct_logit': om.measure_correct_logit,
                         'logit_margin': om.measure_logit_margin,
                         'highest_incorrect_logit': om.measure_highest_incorrect_logit,
                         'accuracy': om.measure_accuracy,
                         'cross_entropy': om.measure_cross_entropy}

def map_nested_dicts(ob, func):
  """ applies func to all the leaves in ob """
  if isinstance(ob, collections.Mapping):
    return {k: map_nested_dicts(v, func) for k, v in ob.items()}
  else:
    return func(ob)

def zero_model_gradients(model):
  for p in model.parameters():
    if p.grad is not None:
      p.grad.data.zero_()

def measure_model_characteristics(msmts_to_make, model, parameter_names, initial_parameters):

  msmts = {}
  for parameter_name, parameter_kwargs in parameter_names.items():
    msmts[parameter_name] = {}
    for msmt in msmts_to_make:
      msmts[parameter_name][msmt] = measurement_functions[msmt](None,
                                               model.named_parameters(),
                                               None,
                                               initial_parameters,
                                               parameter_name,
                                               msmt_type='param',
                                               **parameter_kwargs)
  return msmts

def measure_on_dataset(msmts_to_make, model, dataloader, device, initial_parameters):
  " This is for functions that do not require gradient info "

  model.eval()
  model = model.to(device)

  measured_cumulants = {msmt: MomentAggregator(max_moment=2) for msmt in msmts_to_make}

  for step, (data, targets) in enumerate(dataloader):

    data = data.to(device)
    targets = targets.to(device)
    outputs = model(data)

    for msmt in msmts_to_make:
      sample_wise_msmts = measurement_functions[msmt](outputs,
                                                      model.parameters(),
                                                      targets,
                                                      initial_parameters,
                                                      None,
                                                      None)
      measured_cumulants[msmt].update(sample_wise_msmts)

  measured_cumulants = map_nested_dicts(measured_cumulants,
                                        get_cumulants)

  return measured_cumulants

def measure_characteristic_on_dataset(msmts_to_make,
                                      model,
                                      dataloader,
                                      device,
                                      parameters_to_study,
                                      initial_parameters):
  """ This is for functions which do require the gradient"""
  model.eval()
  model = model.to(device)

  measured_cumulants = {c: {m: MomentAggregator(max_moment=2) for m in msmts_to_make}
                        for c in parameters_to_study}

  for step, (data, targets) in enumerate(dataloader):
    zero_model_gradients(model)

    data = data.to(device)
    targets = targets.to(device)
    outputs = model(data)

    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    loss = loss_fn(outputs, targets)
    loss.backward()


    for msmt in msmts_to_make:
      for parameter_name, parameter_kwargs in parameters_to_study.items():
        sample_wise_msmts = measurement_functions[msmt](None,
                                                 model.named_parameters(),
                                                 None,
                                                 initial_parameters,
                                                 parameter_name,
                                                 msmt_type='grad',
                                                 **parameter_kwargs)
        measured_cumulants[parameter_name][msmt].update(sample_wise_msmts)

  measured_cumulants = map_nested_dicts(measured_cumulants,
                                        get_cumulants)

  return measured_cumulants
