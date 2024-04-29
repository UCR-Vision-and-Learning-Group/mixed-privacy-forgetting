import torch
import torch.nn as nn
from torch.optim import Adam

from torchvision.models import resnet18, ResNet18_Weights

import os
import argparse

from dataset import split_dataset_core_train, get_core_train_loader


def train_core_dataset(core_loader, dataset_id, arch_id, criterion_id, optimizer_id, lr, num_epoch,
                       save_path=None, device_id=0, use_pretrained=True, print_loss_log=10):
    device = 'cuda:{}'.format(device_id) if torch.cuda.is_available() else 'cpu'
    
    # init model
    model = None
    if arch_id == 'resnet18':
        if use_pretrained:
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            model = resnet18()

    if dataset_id == 'cifar10':
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)

    model = model.to(device)

    # init criterion
    criterion = None
    if criterion_id == 'ce':
        criterion = nn.CrossEntropyLoss()
    
    # init optimizer
    optimizer = None
    if optimizer_id == 'adam':
        optimizer = Adam(model.parameters(), lr=lr)
    
    # train
    running_loss = []
    running_acc = []
    for epoch in range(num_epoch):
        model.train()
        for iter_idx, (core_data, core_label) in enumerate(core_loader):
            core_data, core_label = core_data.to(device), core_label.to(device)

            optimizer.zero_grad()
            preds = model(core_data)
            loss = criterion(preds, core_label)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())

            if iter_idx == 0 or (iter_idx + 1) % print_loss_log == 0 or (iter_idx + 1) == len(core_loader):
                print('#########epoch: {}, iter: {}, loss: {}#########'.format(epoch + 1, iter_idx + 1, loss.item())) 

        model.eval()
        with torch.no_grad():
            true_count = 0
            sample_count = 0
            for core_data, core_label in core_loader:
                core_data, core_label = core_data.to(device), core_label.to(device)
                preds = model(core_data)
                
                predicted_label = torch.argmax(preds, dim=1)
                true_count += torch.count_nonzero(predicted_label == core_label).item()
                sample_count += core_data.shape[0]

            print('=========epoch: {}, accuracy: {}========='.format(epoch + 1, (true_count / sample_count)))
            running_acc.append((true_count / sample_count))
    
        if save_path:
            if not os.path.exists(os.path.split(save_path)[0]):
                os.makedirs(os.path.split(save_path)[0])

            torch.save({
                'epoch': epoch, 
                'running_loss': running_loss,
                'running_acc': running_acc,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, save_path)
            

def train_train_dataset():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--mode', dest='mode', type=str, required=True)

    parser.add_argument('-di', '--dataset-id', dest='dataset_id', type=str, required=True)
    parser.add_argument('-ai', '--arch-id', dest='arch_id', type=str, required=True)
    parser.add_argument('-sr', '--split-rate', dest='split_rate', type=float, required=True)
    parser.add_argument('-bs', '--batch-size', dest='batch_size', type=int, required=True)
    parser.add_argument('-ci', '--criterion-id', dest='criterion_id', type=str, required=True)
    parser.add_argument('-oi', '--optimizer-id', dest='optimizer_id', type=str, required=True)
    parser.add_argument('-lr', '--learning-rate', dest='learning_rate', type=float, required=True)
    parser.add_argument('-ne', '--num-epoch', dest='num_epoch', type=int, required=True)

    parser.add_argument('-s', '--shuffle', dest='shuffle', action='store_true')
    parser.add_argument('-sp', '--save-path', dest='save_path', type=str)
    parser.add_argument('-dei', '--device-id', dest='device_id', type=int, default=0)
    parser.add_argument('-up', '--use-pretrained', dest='use_pretrained', action='store_true')
    parser.add_argument('-pll', '--print-loss-log', dest='print_loss_log', default=10)

    args = parser.parse_args()

    if args.mode == 'train-core-dataset':
        core_dataset, _ = split_dataset_core_train(args.dataset_id, args.arch_id, args.split_rate)
        core_loader, _ = get_core_train_loader(core_dataset, _, args.batch_size, shuffle=args.shuffle)
        train_core_dataset(core_loader, args.dataset_id, args.arch_id, args.criterion_id, args.optimizer_id,
                           args.learning_rate, args.num_epoch, save_path=args.save_path, device_id=args.device_id,
                           use_pretrained=args.use_pretrained, print_loss_log=args.print_loss_log)
