import torch
from model import get_core_model_params, get_trained_linear, init_pretrained_model, split_model_to_feature_linear, freeze, thaw
import torch.nn as nn
from train import test_mixed_linear
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
from dataset import get_user_loader

from torch.func import functional_call
import torch.autograd.forward_ad as fwAD

from loss import L2Regularization, LossWrapper
from utils import params_to_device

from torch.optim import SGD
from dataset import split_user_train_dataset_to_remaining_forget, get_remaining_forget_loader

from utils import init_exp
import os
import logging

name_arr = ['resnet50', 'cifar10', 'last{}'.format(5), 'split{}'.format(0.1)]
exp_path = init_exp('forgetting', name_arr)

# loading core model and linearized model -- init in cpu
pretrained_model = init_pretrained_model('resnet50', 'cifar10')
_, linearized_head_core, __ = split_model_to_feature_linear(pretrained_model, 5, None, send_params_to_device=False)
core_model_state_dict = get_core_model_params('checkpoint/05042024-213334-train-user-data-resnet50-cifar10-last5/05042024_213334_train_user_data_resnet50_cifar10_last5_core_model.pth', 'cpu')
feature_backbone, mixed_linear = get_trained_linear('checkpoint/05042024-213334-train-user-data-resnet50-cifar10-last5/05042024_213334_train_user_data_resnet50_cifar10_last5.pth', 'resnet50', 'cifar10', 5)
del _
del __
# _, test_loader = get_user_loader('cifar10', 'resnet50', 256)
# test_mixed_linear(mixed_linear, test_loader, feature_backbone, core_model_state_dict, None, None, 0, device, None, None, None, save_param=False)

v_param = {key: torch.randn_like(value, device='cpu') for key, value in core_model_state_dict.items()} ## init in cpu

class JVPNormLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, feature_backbone, arch, primals, tangents, inp):
        with torch.no_grad():
            inp = feature_backbone(inp)

        dual_params = {}
        with fwAD.dual_level():
            for name, p in primals.items():
                dual_params[name] = fwAD.make_dual(p, tangents[name])
            out = functional_call(arch, dual_params, inp)
            jvp = fwAD.unpack_dual(out).tangent
        return (torch.norm(jvp) ** 2) / inp.shape[0]

def calculate_gradient(feature_backbone, core_model_state_dict, model, loss_fnc, data_loader, device):
    grads = [torch.zeros_like(param) for param in model.parameters()]
    sample_count = 0
    thaw(model)
    for iter_idx, (inp, target) in enumerate(data_loader):
        model.zero_grad()
        inp = inp.to(device)
        target = 5 * target.to(device)
        curr_loss = loss_fnc(model(feature_backbone, core_model_state_dict, inp), target, model.parameters())
        curr_loss.backward()
        for idx, param in enumerate(model.parameters()):
            grads[idx] += (param.grad * inp.shape[0])
        sample_count += inp.shape[0]
        if iter_idx == 0 or (iter_idx + 1) % 50 == 0 or (iter_idx + 1) == len(data_loader):
            print('iter: {}/{}'.format(iter_idx + 1, len(data_loader)))
            logging.info('iter: {}/{}'.format(iter_idx + 1, len(data_loader)))
    freeze(model)
    
    last = []
    for grad in grads:
        tmp = grad / sample_count
        tmp.requires_grad = False
        last.append(tmp)
    return last   
    
class GradientVectorInnerProduct(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, grads, vector_values):
        grad_vector_inner_product_sum = None
        for param, vector_value in zip(grads, vector_values):
            if grad_vector_inner_product_sum is None:
                grad_vector_inner_product_sum = torch.sum(param * vector_value)
            else:
                grad_vector_inner_product_sum += torch.sum(param * vector_value)       
        return grad_vector_inner_product_sum
    
feature_backbone = feature_backbone.to(device)
freeze(feature_backbone)

mixed_linear = mixed_linear.to(device)
freeze(mixed_linear)

linearized_head_core = linearized_head_core.to(device)
freeze(linearized_head_core)

core_model_state_dict = params_to_device(core_model_state_dict, device)
v_param = params_to_device(v_param, device)
for param in v_param.values():
    param.requires_grad = True

remaining_dataset, forget_dataset = split_user_train_dataset_to_remaining_forget('cifar10', 'resnet50', 0.1)
remain_loader, forget_loader = get_remaining_forget_loader(remaining_dataset, forget_dataset, 256)

main_criterion = LossWrapper([nn.MSELoss(), L2Regularization()], [1, 0.0005])
grads = calculate_gradient(feature_backbone, core_model_state_dict, mixed_linear, main_criterion, remain_loader, device)

grad_path = os.path.split(exp_path)[0]
grad_path = os.path.join(grad_path, '{}_grads.pth'.format('_'.join(os.path.split(grad_path)[1].split('-'))))
torch.save({
    'grads': grads,
}, grad_path)

jvp_norm_criterion = JVPNormLoss()
gradient_vector_inner_product_criterion = GradientVectorInnerProduct()
regularizor_criterion = L2Regularization()


optimizer = SGD(v_param.values(), lr=0.001, momentum=0.999)

for epoch in range(150):
    if (epoch + 1) in [30, 60, 90, 120]:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5
    for iter_idx, (data, label) in enumerate(remain_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        jvp_norm_loss = 0.5 * jvp_norm_criterion(feature_backbone, linearized_head_core, core_model_state_dict, v_param, data)
        gradient_vector_inner_product_loss = gradient_vector_inner_product_criterion(grads, v_param.values())
        regularizor_loss = 0.5 * 0.0005 * regularizor_criterion(v_param.values())
        loss = jvp_norm_loss + regularizor_loss - gradient_vector_inner_product_loss
        loss.backward()
        optimizer.step()
        if iter_idx == 0 or (iter_idx + 1) % 50 == 0 or (iter_idx + 1) == len(remain_loader):
            print('epoch: {}/{}, iter: {}/{}, loss: {}'.format(epoch + 1, 3, iter_idx + 1, len(remain_loader), loss.item()))
            logging.info('epoch: {}/{}, iter: {}/{}, loss: {}'.format(epoch + 1, 3, iter_idx + 1, len(remain_loader), loss.item()))

v_param_path = os.path.split(exp_path)[0]
v_param_path = os.path.join(v_param_path, '{}_v_param.pth'.format('_'.join(os.path.split(v_param_path)[1].split('-'))))
torch.save({
    'v_param': v_param,
}, v_param_path)

forgetted = {name: first - second for name, first, second in zip(mixed_linear.tangents.keys(), mixed_linear.tangents.values(), v_param.values())}

with torch.no_grad():
    state_dict = mixed_linear.state_dict()
    for name, value in forgetted.items():
        state_dict['tangent_model.{}'.format(name)] = value
    mixed_linear.load_state_dict(state_dict)

_, test_loader = get_user_loader('cifar10', 'resnet50', 256, shuffle=False)
test_mixed_linear(mixed_linear, test_loader, feature_backbone, core_model_state_dict, None, None, 0, device, None, None, None, save_param=False)

torch.save({
    'model_state_dict': mixed_linear.state_dict(),
}, exp_path)