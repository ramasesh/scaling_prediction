import torch
from src import utils
from src.utils import AverageMeter
import time
import collections
import numpy as np

def test(model, criterion, test_loader, run_config):
    device = torch.device(run_config['device'])

    model.eval()

    loss_meter = AverageMeter()
    correct_meter = AverageMeter()
    start = time.time()
    with torch.no_grad():
        for step, (data, targets) in enumerate(test_loader):
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)

            _, preds = torch.max(outputs, dim=1)

            loss_ = loss.item()
            correct_ = preds.eq(targets).sum().item()
            num = data.size(0)

            loss_meter.update(loss_, num)
            correct_meter.update(correct_, 1)

        accuracy = correct_meter.sum / len(test_loader.dataset)

        elapsed = time.time() - start

    test_log = collections.OrderedDict({
                'loss': loss_meter.avg,
                'accuracy': accuracy,
                'time': elapsed})
    return test_log

def train(model, optimizer, scheduler, criterion, train_loader, run_config):

    device = torch.device(run_config['device'])

    for param_group in optimizer.param_groups:
      current_lr = param_group['lr']

    model.train()

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    start = time.time()

    for step, (data, targets) in enumerate(train_loader):

        if torch.cuda.device_count() == 1:
            data = data.to(device)
            targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        loss_ = loss.item()
        num = data.size(0)
        accuracy = utils.accuracy(outputs, targets)[0].item()

        loss_meter.update(loss_, num)
        accuracy_meter.update(accuracy, num)

        if scheduler is not None:
            scheduler.step()

    elapsed = time.time() - start

    train_log = collections.OrderedDict({
                    'loss': loss_meter.avg,
                    'accuracy': accuracy_meter.avg,
                    'time': elapsed})
    return train_log

def update_state(state, epoch, accuracy, model, optimizer):
    state['state_dict'] = model.state_dict()
    state['optimizer'] = optimizer.state_dict()
    state['epoch'] = epoch
    state['accuracy'] = accuracy

    # update best accuracy
    if accuracy > state['best_accuracy']:
        state['best_accuracy'] = accuracy
        state['best_epoch'] = epoch

    return state
