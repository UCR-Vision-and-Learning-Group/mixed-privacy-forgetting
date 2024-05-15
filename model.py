import torch
import torch.nn as nn
from torch.func import functional_call
import torch.autograd.forward_ad as fwAD
from torchvision.models import resnet50, ResNet50_Weights

from utils import params_to_device
import logging


def reset_parameters(arch):
    for kid in arch.children():
        if len([grand_kid for grand_kid in kid.children()]) == 0:
            if hasattr(kid, 'reset_parameters'):
                kid.reset_parameters()
        else:
            reset_parameters(kid)

def freeze(arch):
    for p in arch.parameters():
        p.requires_grad = False

def thaw(arch):
    for p in arch.parameters():
        p.requires_grad = True

class Flatten(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return torch.flatten(x, 1)

def init_pretrained_model(arch_id, dataset_id, use_default=True, pretrained_model_path=None, hidden_layers=None):
    # hidden layers --> is an array of number of perceptrons in each layer hidden
    pretrained_model = None
    if arch_id == 'resnet50':
        if use_default and pretrained_model_path is None:
            pretrained_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            checkpoint = torch.load(pretrained_model_path)
            pretrained_model = resnet50()
        
        if dataset_id == 'cifar10':
            num_ftrs = pretrained_model.fc.in_features
            if hidden_layers is None:
                pretrained_model.fc = nn.Linear(num_ftrs, 10, bias=False)
            else:
                logging.info('arbitrary hidden layers are added {}'.format(hidden_layers))
                kid_arr = []
                for kid in pretrained_model.children():
                    grand_kid_arr = [c for c in kid.children()]
                    if len(grand_kid_arr) > 0:
                        for grand_kid in grand_kid_arr:
                            kid_arr.append(grand_kid)
                    else:
                        kid_arr.append(kid)
                kid_arr = kid_arr[:-1] + [Flatten()]
                curr_in = num_ftrs
                for hidden_idx in range(len(hidden_layers)):
                    kid_arr.append(nn.Linear(curr_in, hidden_layers[hidden_idx]))
                    curr_in = hidden_layers[hidden_idx]
                kid_arr.append(nn.Linear(curr_in, 10, bias=False))
                pretrained_model = nn.Sequential(*kid_arr)

        if pretrained_model_path is not None:
            pretrained_model.load_state_dict(checkpoint['model_state_dict'])

    return pretrained_model

def split_model_to_feature_linear(pretrained_model, number_of_linearized_components, device, send_params_to_device=True):
    kid_arr = []
    for kid in pretrained_model.children():
        grand_kid_arr = [c for c in kid.children()]
        if len(grand_kid_arr) > 0:
            for grand_kid in grand_kid_arr:
                kid_arr.append(grand_kid)
        else:
            kid_arr.append(kid)

    feature_backbone = nn.Sequential(*kid_arr[:-number_of_linearized_components])
    freeze(feature_backbone)
    if send_params_to_device:
        feature_backbone = feature_backbone.to(device)

    linearized_head_core = kid_arr[-number_of_linearized_components:]
    linearized_head_core.insert(len(linearized_head_core) - 1, Flatten())

    linearized_head_core = nn.Sequential(*linearized_head_core)
    params = {name: p.detach().clone() for name, p in linearized_head_core.named_parameters()}
    if send_params_to_device:
        params = params_to_device(params, device)
    return feature_backbone, linearized_head_core, params

class MixedLinear(nn.Module):
    def __init__(self, arch) -> None:
        super().__init__()
        self.tangent_model = arch
        reset_parameters(self.tangent_model)
        thaw(self.tangent_model)
        self.tangents = {name: p for name, p in self.tangent_model.named_parameters()}

    def forward(self, feature_backbone, core_model_params, inp):
        with torch.no_grad():
            inp = feature_backbone(inp)

        dual_params = {}
        with fwAD.dual_level():
            for name, p in core_model_params.items():
                dual_params[name] = fwAD.make_dual(p, self.tangents[name])
            out = functional_call(self.tangent_model, dual_params, inp)
            jvp = fwAD.unpack_dual(out).tangent
        return out + jvp

    def set_params(self, named_params):
        # set only named params
        pass

    
def get_core_model_params(core_model_params_path, device):
    core_model_state_dict = torch.load(core_model_params_path)
    core_model_state_dict = core_model_state_dict['params']
    return params_to_device(core_model_state_dict, device)

def get_trained_linear(checkpoint_path, arch_id, dataset_id, number_of_linearized_components):
    checkpoint = torch.load(checkpoint_path)
    pretrained_model = init_pretrained_model(arch_id, dataset_id)
    feature_backbone, linearized_head_core, __ = split_model_to_feature_linear(pretrained_model, number_of_linearized_components, None, send_params_to_device=False)
    mixed_linear = MixedLinear(linearized_head_core)
    mixed_linear.load_state_dict(checkpoint['model_state_dict'])
    return feature_backbone, mixed_linear