import torch.nn as nn
import torch
from torch.func import functional_call
import torch.autograd.forward_ad as fwAD


# TODO: the implementations are not that generic might be better than that also the parts
#  implemented for forgetting should be added here
class L2Regularization(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.need_params = True

    # noinspection PyMethodMayBeStatic
    def forward(self, params):
        l2_loss = None
        for param in params:
            if l2_loss is None:
                l2_loss = param.norm(2) ** 2
            else:
                l2_loss = l2_loss + (param.norm(2) ** 2)
        return l2_loss / 2


class MSELossDiv2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, inp, target):
        loss = self.criterion(inp, target)
        return loss / 2


class LossWrapper(nn.Module):
    def __init__(self, loss_modules, hyperparameters):
        super().__init__()
        self.loss_modules = loss_modules
        self.hyperparameters = hyperparameters

    def forward(self, inp, target, params):
        loss = None
        for loss_module, hyperparameter in zip(self.loss_modules, self.hyperparameters):
            if loss is None:
                if hasattr(loss_module, 'need_params'):
                    loss = loss_module(params) * hyperparameter
                else:
                    loss = loss_module(inp, target) * hyperparameter
            else:
                if hasattr(loss_module, 'need_params'):
                    loss += loss_module(params) * hyperparameter
                else:
                    loss += loss_module(inp, target) * hyperparameter

        return loss


# losses observed from mixed privacy paper
class JVPNormLoss(nn.Module):
    def __init__(self, activation_variant=False) -> None:
        super().__init__()
        self.activation_variant = activation_variant

    # noinspection PyMethodMayBeStatic
    def forward(self, feature_backbone, arch, primals, tangents, inp):
        if not self.activation_variant:
            with torch.no_grad():
                inp = feature_backbone(inp)

        dual_params = {}
        with fwAD.dual_level():
            for name, p in primals.items():
                dual_params[name] = fwAD.make_dual(p, tangents[name])
            out = functional_call(arch, dual_params, inp)
            jvp = fwAD.unpack_dual(out).tangent
        return (torch.norm(jvp) ** 2) / inp.shape[0]


class GradientVectorInnerProduct(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, grads, vector_values):
        grad_vector_inner_product_sum = None
        for param, vector_value in zip(grads, vector_values):
            if grad_vector_inner_product_sum is None:
                grad_vector_inner_product_sum = torch.sum(param * vector_value)
            else:
                grad_vector_inner_product_sum += torch.sum(param * vector_value)
        return grad_vector_inner_product_sum
