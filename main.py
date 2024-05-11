import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler, Adam

import argparse
import numpy as np
import random

from utils import *
from train import *
from loss import *
from dataset import *
from model import *

def set_deterministic_environment(seed=13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def train_user_data(arch_id, dataset_id, number_of_linearized_components,
                    use_default=True, pretrained_model_path=None,
                    device_id=0, shuffle=True, split_rate=0, weight_decay=0.0005):
    
    name_arr = [arch_id, dataset_id, 'last{}'.format(number_of_linearized_components)]
    if split_rate > 0:
        name_arr = name_arr + ['split{}'.format(split_rate)]

    exp_path = init_exp('train-user-data', name_arr)
    
    device = 'cuda:{}'.format(device_id) if torch.cuda.is_available() else 'cpu'
    pretrained_model = init_pretrained_model(arch_id, dataset_id, use_default=use_default, pretrained_model_path=pretrained_model_path)

    feature_model, linear_model, linear_model_params = split_model_to_feature_linear(pretrained_model,
                                                                                     number_of_linearized_components,
                                                                                     device)
    torch.save({
        'params': linear_model_params
    }, get_core_model_path(exp_path))

    mixed_linear_model = MixedLinear(linear_model)
    mixed_linear_model = mixed_linear_model.to(device)

    criterion = LossWrapper([MSELossDiv2(), L2Regularization()], [1, weight_decay])
    optimizer = SGD(mixed_linear_model.parameters(), lr=0.05, momentum=0.9)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[24, 39], gamma=0.1)

    if pretrained_model_path is not None:
        core_dataset, user_train_dataset, user_test_dataset = split_dataset_to_core_user(dataset_id, arch_id, split_rate, seed=13)
        _, user_train_loader, user_test_loader = get_core_user_loader(core_dataset, user_train_dataset, user_test_dataset, 64, shuffle=shuffle)
    elif split_rate > 0:
        remaining_dataset, _ = split_user_train_dataset_to_remaining_forget(dataset_id, arch_id, split_rate, seed=13)
        user_train_loader, _ = get_remaining_forget_loader(remaining_dataset, _, 64, shuffle=True)
        _, user_test_loader = get_user_loader(dataset_id, arch_id, 64, shuffle=shuffle)
    else:
        user_train_loader, user_test_loader = get_user_loader(dataset_id, arch_id, 64, shuffle=shuffle)


    running_loss = []
    running_test_acc = []
    running_train_acc = []

    best_model_test_acc = -1
    best_model_epoch = -1

    init_checkpoint(running_loss, running_test_acc, running_train_acc, best_model_test_acc, best_model_epoch, exp_path)

    for epoch in range(50):
        checkpoint = get_checkpoint(exp_path)
        running_test_acc, checkpoint = test_mixed_linear(mixed_linear_model, user_test_loader, feature_model,
                                                         linear_model_params, optimizer, running_test_acc, epoch, device,
                                                         checkpoint, best_model_test_acc, best_model_epoch)
        running_train_acc, checkpoint = train_accuracy_mixed_linear(mixed_linear_model, user_train_loader, feature_model,
                                                                    linear_model_params, running_train_acc, epoch, device,
                                                                    checkpoint)
        mixed_linear_model, optimizer, scheduler, running_loss, checkpoint = train_mixed_linear(mixed_linear_model, user_train_loader,
                                                                                                feature_model, linear_model_params,
                                                                                                optimizer, criterion, scheduler,
                                                                                                running_loss, device, epoch, checkpoint)
        set_checkpoint(checkpoint, exp_path)

    checkpoint = get_checkpoint(exp_path)
    running_test_acc, checkpoint = test_mixed_linear(mixed_linear_model, user_test_loader, feature_model,
                                                        linear_model_params, optimizer, running_test_acc, epoch, device,
                                                        checkpoint, best_model_test_acc, best_model_epoch)
    running_train_acc, checkpoint = train_accuracy_mixed_linear(mixed_linear_model, user_train_loader, feature_model,
                                                                linear_model_params, running_train_acc, epoch, device,
                                                                checkpoint)
    set_checkpoint(checkpoint, exp_path)

def pretrain(arch_id, dataset_id, split_rate, device_id=0, shuffle=True):
    exp_path = init_exp('pretrain', [arch_id, dataset_id, 'split{}'.format(split_rate)])
    
    device = 'cuda:{}'.format(device_id) if torch.cuda.is_available() else 'cpu'
    pretrained_model = init_pretrained_model(arch_id, dataset_id, use_default=True, pretrained_model_path=None)
    pretrained_model = pretrained_model.to(device)

    core_dataset, user_train_dataset, user_test_dataset = split_dataset_to_core_user(dataset_id, arch_id, split_rate, seed=13)
    core_train_loader, _, user_test_loader = get_core_user_loader(core_dataset, user_train_dataset, user_test_dataset, 64, shuffle=shuffle)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(pretrained_model.parameters(), lr=0.001)

    running_loss = []
    running_test_acc = []
    running_train_acc = []

    best_model_test_acc = -1
    best_model_epoch = -1

    init_checkpoint(running_loss, running_test_acc, running_train_acc, best_model_test_acc, best_model_epoch, exp_path)

    for epoch in range(20):
        checkpoint = get_checkpoint(exp_path)
        running_test_acc, checkpoint = test_pretrain(pretrained_model, user_test_loader, optimizer,
                                                     running_test_acc, epoch, device, checkpoint,
                                                     best_model_test_acc, best_model_epoch)
        
        pretrained_model, optimizer, running_loss, checkpoint = train_pretrain(pretrained_model, core_train_loader,
                                                                               optimizer, criterion, running_loss,
                                                                               device, epoch, checkpoint)
        set_checkpoint(checkpoint, exp_path)

    checkpoint = get_checkpoint(exp_path)
    running_test_acc, checkpoint = test_pretrain(pretrained_model, user_test_loader, optimizer,
                                                running_test_acc, epoch, device, checkpoint,
                                                best_model_test_acc, best_model_epoch)
    
    set_checkpoint(checkpoint, exp_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--mode', dest='mode', type=str, required=True)

    parser.add_argument('-di', '--dataset-id', dest='dataset_id', type=str, required=True)
    parser.add_argument('-ai', '--arch-id', dest='arch_id', type=str, required=True)
    parser.add_argument('-nlc', '--number-of-linearized-components', dest='number_of_linearized_components', type=int)

    parser.add_argument('-dei', '--device-id', dest='device_id', type=int, default=0)
    parser.add_argument('-ud', '--use-default', dest='use_default', action='store_true')
    parser.add_argument('-pmp', '--pretrained-model-path', dest='pretrained_model_path', type=str)
    parser.add_argument('-sr', '--split-rate', dest='split_rate', type=float, default=0) # TODO: handle the exceptions

    parser.add_argument('-wd', '--weight-decay', dest='weight_decay', type=float, default=0.0005)

    args = parser.parse_args()

    set_deterministic_environment()

    if args.mode == 'train-user-data':
        train_user_data(args.arch_id, args.dataset_id, args.number_of_linearized_components,
                        use_default=args.use_default, pretrained_model_path=args.pretrained_model_path,
                        device_id=args.device_id, split_rate=args.split_rate)
    elif args.mode == 'pretrain':
        pretrain(args.arch_id, args.dataset_id, args.split_rate, device_id=args.device_id, shuffle=True)