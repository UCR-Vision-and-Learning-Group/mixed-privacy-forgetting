import torch
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Lambda

import numpy as np


def get_data_transformations(dataset_id, arch_id, one_hot=True):
    transform, target_transform = None, None
    if dataset_id == 'cifar10' and arch_id == 'resnet18':
        transform = Compose([
            Resize((224, 224)),  # Resize to match ResNet input size
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # Normalize
        ])

        if one_hot:
            target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    
    return transform, target_transform


class SubsetToDataset(Dataset):
    def __init__(self, subset, transform=None, target_transform=None):
        self.subset = subset
        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)

        if self.target_transform:
            y = self.target_transform(y)

        return x, y
        
    def __len__(self):
        return len(self.subset)


def split_dataset_core_train(dataset_id, arch_id, split_rate):
    whole_dataset, transform, target_transform = None, None, None
    if dataset_id == 'cifar10':
        whole_dataset = CIFAR10('./data', train=True, transform=None, target_transform=None, download=True)
        transform, target_transform = get_data_transformations(dataset_id, arch_id)
    
    len_train_dataset = int(len(whole_dataset) * split_rate)
    len_core_dataset = len(whole_dataset) - len_train_dataset
    core_subset, train_subset = random_split(whole_dataset, [len_core_dataset, len_train_dataset])    

    core_dataset = SubsetToDataset(core_subset, transform=transform, target_transform=None) # ce loss
    train_dataset = SubsetToDataset(train_subset, transform=transform, target_transform=target_transform) # mse loss
    return core_dataset, train_dataset

def get_core_train_loader(core_dataset, train_dataset, batch_size, shuffle=True):
    core_loader = DataLoader(core_dataset, batch_size=batch_size, shuffle=shuffle)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    return core_loader, train_loader


def split_train_dataset_remaining_forget():
    pass