import torch
import torch.nn as nn
from torch.func import functional_call
import torch.autograd.forward_ad as fwAD

from torchvision.models import resnet18

class SimpleNet(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.lin_1 = nn.Linear(in_features, 128)
        self.lin_2 = nn.Linear(128, out_features)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.lin_2(self.act(self.lin_1(x)))

class MixedLinearModel(nn.Module):
    def __init__(self, dataset_id, arch_id, zero_init=False):
        super().__init__()
        self.tangent_model = None
        self.dataset_id = dataset_id
        self.arch_id = arch_id
        self.zero_init = zero_init
        if self.arch_id == 'resnet18':
            self.tangent_model = resnet18()
            if self.dataset_id == 'cifar10':
                num_ftrs = self.tangent_model.fc.in_features
                self.tangent_model.fc = nn.Linear(num_ftrs, 10)
        elif self.arch_id == 'simple':
            if self.dataset_id == 'mnist':
                self.tangent_model = SimpleNet(784, 10)

        if self.zero_init:
            for p in self.tangent_model.parameters():
                p.data.fill_(0)

        self.tangents = {name: p for name, p in self.tangent_model.named_parameters()}
    
    def forward(self, core_model_params, inp):
        dual_params = {}
        with fwAD.dual_level():
            for name, p in core_model_params.items():
                dual_params[name] = fwAD.make_dual(p, self.tangents[name])
            out = functional_call(self.tangent_model, dual_params, inp)
            jvp = fwAD.unpack_dual(out).tangent
        return out + jvp