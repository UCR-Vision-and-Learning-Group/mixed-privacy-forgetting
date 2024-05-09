import torch

import logging
from datetime import datetime
import os
import matplotlib.pyplot as plt

def init_exp(mode, name_arr):
    curr_file_name = mode.split('-')
    curr_file_name = curr_file_name + name_arr
    now = datetime.now()
    folder_name_datetime = now.strftime('%m%d%Y-%H%M%S')
    file_name_datetime = now.strftime('%m%d%Y_%H%M%S')
    folder_name_exp = '-'.join(curr_file_name)
    file_name_exp = '_'.join(curr_file_name)
    folder_name_exp = '{}-{}'.format(folder_name_datetime, folder_name_exp)
    file_name_exp = '{}_{}'.format(file_name_datetime, file_name_exp)

    exp_dir = os.path.join('./checkpoint/', folder_name_exp)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    log_path = os.path.join(exp_dir, '{}.log'.format(file_name_exp))
    logging.basicConfig(filename=log_path, level=logging.INFO)

    logging.info('experiment files initialized: {}'.format(exp_dir))
    exp_path = os.path.join(exp_dir, '{}.pth'.format(file_name_exp))
    return exp_path


def get_core_model_path(exp_path):
    split_exp_path = os.path.split(exp_path)
    return os.path.join(split_exp_path[0], '{}_core_model.pth'.format('.'.join(split_exp_path[1].split('.')[:-1])))

def init_checkpoint(running_loss, running_test_acc, running_train_acc, best_model_test_acc, best_model_epoch, exp_path):
    torch.save({
        'running_loss': running_loss,
        'running_test_acc': running_test_acc,
        'running_train_acc': running_train_acc,
        'best_model_test_acc': best_model_test_acc,
        'best_model_epoch': best_model_epoch,
        'model_state_dict': None,
        'optimizer_state_dict': None,
        'core_model_params': None
    }, exp_path)

def get_checkpoint(exp_path):
    return torch.load(exp_path)

def set_checkpoint(checkpoint, exp_path):
    torch.save(checkpoint, exp_path)

def params_to_device(param, device):
    return {key: value.to(device) for key, value in param.items()}

# plotting functions
def plot_everything():
    # loss
    pass