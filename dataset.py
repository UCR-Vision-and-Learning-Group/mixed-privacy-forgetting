import torch
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Lambda

import os


# TODO: there are lots of buggy implementation becuase of time constraints,
#  needs to be checked after observing the experiments

def get_data_transformations(dataset_id, arch_id, one_hot=True):
    transform, target_transform = None, None
    if dataset_id == 'cifar10' and (arch_id == 'resnet50' or arch_id == 'resnet18'):
        transform = Compose([
            Resize((224, 224)),  # Resize to match ResNet input size
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # Normalize
        ])

        if one_hot:
            target_transform = Lambda(
                lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, y.clone().detach(), value=1))
    elif dataset_id == 'cifar10-act' and (arch_id == 'resnet50' or arch_id == 'resnet18'):
        if one_hot:
            target_transform = Lambda(
                lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, y.clone().detach(), value=1))
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


def split_dataset_to_core_user(dataset_id, arch_id, split_rate, seed=13, number_of_linearized_components=None):
    whole_dataset, transform, target_transform, user_test_dataset = None, None, None, None
    if dataset_id == 'cifar10':
        whole_dataset = CIFAR10('./data', train=True, transform=None, target_transform=None, download=True)
        transform, target_transform = get_data_transformations(dataset_id, arch_id)
        user_test_dataset = CIFAR10('./data', train=False, transform=transform, download=True)
    elif dataset_id == 'cifar10-act':
        root_path = './data/{}-{}-last{}'.format(arch_id, dataset_id.split('-')[0], number_of_linearized_components)
        whole_dataset = ActivationDataset(root_path, train=True, transform=None, target_transform=None)
        transform, target_transform = get_data_transformations(dataset_id, arch_id)
        user_test_dataset = ActivationDataset(root_path, train=False, transform=transform)

    len_user_train_dataset = int(len(whole_dataset) * split_rate)
    len_core_dataset = len(whole_dataset) - len_user_train_dataset
    generator = torch.Generator().manual_seed(seed)
    core_subset, user_train_subset = random_split(whole_dataset, [len_core_dataset, len_user_train_dataset],
                                                  generator=generator)

    core_dataset = SubsetToDataset(core_subset, transform=transform, target_transform=None)  # ce loss
    user_train_dataset = SubsetToDataset(user_train_subset, transform=transform,
                                         target_transform=target_transform)  # mse loss
    return core_dataset, user_train_dataset, user_test_dataset


def get_core_user_loader(core_dataset, train_dataset, test_dataset, batch_size, shuffle=True):
    core_loader = DataLoader(core_dataset, batch_size=batch_size, shuffle=shuffle)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return core_loader, train_loader, test_loader


def get_user_loader(dataset_id, arch_id, batch_size, shuffle=True, number_of_linearized_components=None):
    user_train_dataset = None
    user_test_dataset = None
    if dataset_id == 'cifar10':
        if arch_id == 'resnet50' or arch_id == 'resnet18':
            transform, target_transform = get_data_transformations(dataset_id, arch_id)
            user_train_dataset = CIFAR10('./data', train=True, transform=transform, target_transform=target_transform,
                                         download=True)
            user_test_dataset = CIFAR10('./data', train=False, transform=transform, download=True)
    elif dataset_id == 'cifar10-act':
        root_path = './data/{}-{}-last{}'.format(arch_id, dataset_id.split('-')[0], number_of_linearized_components)
        transform, target_transform = get_data_transformations(dataset_id, arch_id)
        user_train_dataset = ActivationDataset(root_path, train=True, transform=transform,
                                               target_transform=target_transform)
        user_test_dataset = ActivationDataset(root_path, train=False, transform=transform)
    user_train_loader = DataLoader(user_train_dataset, batch_size=batch_size, shuffle=shuffle)
    user_test_loader = DataLoader(user_test_dataset, batch_size=256, shuffle=shuffle)
    return user_train_loader, user_test_loader


def split_user_train_dataset_to_remaining_forget(dataset_id, arch_id, split_rate, seed=13,
                                                 number_of_linearized_components=None):
    whole_dataset, transform, target_transform = None, None, None
    if dataset_id == 'cifar10':
        whole_dataset = CIFAR10('./data', train=True, transform=None, target_transform=None, download=True)
        transform, target_transform = get_data_transformations(dataset_id, arch_id)
    elif dataset_id == 'cifar10-act':
        root_path = './data/{}-{}-last{}'.format(arch_id, dataset_id.split('-')[0], number_of_linearized_components)
        whole_dataset = ActivationDataset(root_path, train=True, transform=None, target_transform=None)
        transform, target_transform = get_data_transformations(dataset_id, arch_id)

    forget_dataset = int(len(whole_dataset) * split_rate)
    len_remaining_dataset = len(whole_dataset) - forget_dataset
    generator = torch.Generator().manual_seed(seed)
    core_subset, user_train_subset = random_split(whole_dataset, [len_remaining_dataset, forget_dataset],
                                                  generator=generator)

    remaining_dataset = SubsetToDataset(core_subset, transform=transform, target_transform=target_transform)
    forget_dataset = SubsetToDataset(user_train_subset, transform=transform, target_transform=target_transform)
    return remaining_dataset, forget_dataset


def get_remaining_forget_loader(remaining_dataset, forget_dataset, batch_size, shuffle=True):
    remaining_loader = DataLoader(remaining_dataset, batch_size=batch_size, shuffle=shuffle)
    forget_loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=shuffle)
    return remaining_loader, forget_loader


class ActivationDataset(Dataset):
    def __init__(self, root_path, train=True, target_transform=None, transform=None):
        super().__init__()
        self.root_path = root_path
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if self.train:
            data_dict = torch.load(os.path.join(self.root_path, 'train_data.pth'))
            self.data = data_dict['data'].squeeze()
            self.target = data_dict['label']
        else:
            data_dict = torch.load(os.path.join(self.root_path, 'test_data.pth'))
            self.data = data_dict['data'].squeeze()
            self.target = data_dict['label']

    def __getitem__(self, idx):
        target = self.target[idx]
        data = self.data[idx]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self):
        return self.data.shape[0]
