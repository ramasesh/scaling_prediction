import importlib
import copy
import json
import pathlib
import shutil
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from google.cloud import storage
import os
import itertools

from src.models import multitaskmodels as mtm

def str2bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise RuntimeError('Boolean value expected')

def str2list(s):
    """
    converts a string representation of a list
      (with parentheses instead of square-brackets)
    into a list

    acceptable inputs:
      "(4,5)" -> [4,5]
      "((4,5),(6,7))" -> [[4,5],[6,7]]

    Note:
      parentheses were used here to be compatible
      with Caliban.

    """

    s = s.replace('(', '[')
    s = s.replace(')', ']')
    print(s)
    if s[0] == '[' and s[-1] == ']':
      return json.loads(s)
    else:
      raise RuntimeError('List value expected')

def str2splittasklist(s):
    """
    converts a string representation of a task or sequence of tasks
    (defined by a 'split', i.e. a subset of the full set of classes)
    into a list of tasks.

    This is essentially the same method as str2list(), but with some
    additional checking in order to make sure the list returned is a
    valid task specification.

    Further, if only one split is specified, i.e. via the string
    "(4,5)",
    this method will wrap the resulting output so we return
    [[4,5]].
    That is, passing in the arguments
    "(4,5)", and "((4,5))",
    will result in identical output

    """

    task_list = str2list(s)

    # check that we have either a list of ints, or a list of lists
    if not isinstance(task_list, list):
      raise RuntimeError('List value expected')
    else:
      if isinstance(task_list[0], list):
        # if we have a list of lists, the objects in those lists
        # must be lists of ints
        for t in task_list:
          if not isinstance(t, list):
            raise RuntimeError('List of ints or list of lists of ints expected')
          else:
            for ti in t:
              if not (isinstance(ti, int) or isinstance(ti, str)):
                raise RuntimeError('List of ints/str or list of lists of ints/str expected')
      else:
        # if we have a list of primitives, those primitives must be ints
        for t in task_list:
          if not (isinstance(t, int) or isinstance(t, str)):
            raise RuntimeError('List of ints/str or list of lists of ints/str expected')
        task_list = [task_list]

    return task_list

def load_model(config):
    module = importlib.import_module('src.models.{}'.format(config['arch']))
    Network = getattr(module, 'Network')
    return Network(config)

def count_params(model):
    return sum([param.view(-1).size()[0] for param in model.parameters()])

def save_checkpoint(state, outdir):
    model_path = outdir / 'model_state.pth'
    best_model_path = outdir / 'model_best_state.pth'
    torch.save(state, model_path)
    if state['best_epoch'] == state['epoch']:
        shutil.copy(model_path, best_model_path)

def save_epoch_logs(epoch_logs, outdir):
    dirname = outdir.resolve().as_posix().replace('/', '_')
    tempdir = pathlib.Path(tempfile.mkdtemp(prefix=dirname, dir='/tmp'))
    temppath = tempdir / 'log.json'
    with open(temppath, 'w') as fout:
        json.dump(epoch_logs, fout, indent=2)
    shutil.copy(temppath.as_posix(), outdir / temppath.name)
    shutil.rmtree(tempdir, ignore_errors=True)

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count

def _get_optimizer(model_parameters, optim_config):
    optimizer_name = optim_config['optimizer']
    parameters_to_optimize = filter(lambda p: p.requires_grad, model_parameters)
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            parameters_to_optimize,
            lr=optim_config['base_lr'],
            momentum=optim_config['momentum'],
            weight_decay=optim_config['weight_decay'],
            nesterov=optim_config['nesterov'])
    else:
        raise RuntimeError('SGD is currently the only allowed optimizer')
    return optimizer

def _get_scheduler(optimizer, optim_config):
    """" scheduler should be stepped every step, not epoch,
    without knowledge of the epoch """

    if optim_config['scheduler'] == 'multistep':
        step_milestones = [m * optim_config['steps_per_epoch'] for m in optim_config['milestones']]
        def lr_lambda(step):
            count = 0
            for i in step_milestones:
                if i < step:
                    count = count + 1
            lr_multiple = np.power(optim_config['lr_decay'], count)
            return lr_multiple

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda = lr_lambda)
    elif optim_config['scheduler'] == 'cosine':
        total_steps = optim_config['epochs'] * optim_config['steps_per_epoch']

        def lr_lambda(step):
            lr_min = 0.
            return cosine_annealing(step, total_steps, 1., lr_min)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda = lr_lambda)
    else:
        scheduler = None

    return scheduler

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
        1 + np.cos(step / total_steps * np.pi))

def get_criterion(data_config):
    # placeholder to support, e.g., augmentations
    train_criterion = nn.CrossEntropyLoss(reduction='mean')
    test_criterion = nn.CrossEntropyLoss(reduction='mean')
    return train_criterion, test_criterion

def create_optimizer(model_parameters, optim_config):
    optimizer = _get_optimizer(model_parameters, optim_config)
    scheduler = _get_scheduler(optimizer, optim_config)
    return optimizer, scheduler

def accuracy(output, target, topk=(1, )):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1 / batch_size))
    return res


def onehot_encoding(label, n_classes):
    return torch.zeros(label.size(0), n_classes).to(label.device).scatter_(
        1, label.view(-1, 1), 1)

def cross_entropy_loss(input, target, reduction):
    logp = F.log_softmax(input, dim=1)
    loss = torch.sum(-logp * target, dim=1)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(
            '`reduction` must be one of \'none\', \'mean\', or \'sum\'.')

def upload_to_bucket(bucket_name, save_name, outdir):
  """uploads files in the local directory 'outdir' to the
  bucket bucket_name, in the folder save_name"""
  client = storage.Client()
  bucket = client.get_bucket(bucket_name)
  filenames = os.listdir(outdir)

  for filename in filenames:
    blob = bucket.blob(save_name + filename)
    blob.upload_from_filename(str(outdir) + '/' + filename)

def save_files(save_config, args):
  allowed_save_types = ['bucket', 'none']

  if isinstance(save_config, str):
    with open(save_config) as f:
      save_config = json.load(f)

  save_type = save_config['save_type']
  assert save_type in allowed_save_types

  if save_type == 'none':
    return
  elif save_type == 'bucket':
    upload_to_bucket(save_config['bucket_name'],
                     args['save_name'],
                     args['outdir'])

def extract_til(s, c='.'):
  "extracts the contents of string s until character c"
  i = s.index(c)
  return s[:i]
