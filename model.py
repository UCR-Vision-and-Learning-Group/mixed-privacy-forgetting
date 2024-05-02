import torch
import torch.nn as nn
from torch.func import functional_call
import torch.autograd.forward_ad as fwAD


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

def split_model_to_feature_linear(pretrained_model, number_of_linearized_components, device):
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
    feature_backbone = feature_backbone.to(device)

    linearized_head_core = kid_arr[-number_of_linearized_components:]
    linearized_head_core.insert(len(linearized_head_core) - 1, Flatten())

    linearized_head_core = nn.Sequential(*linearized_head_core)
    params = {name: p.detach().clone().to(device) for name, p in linearized_head_core.named_parameters()}

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