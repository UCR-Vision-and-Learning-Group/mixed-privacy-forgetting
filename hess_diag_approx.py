from dataset import get_remaining_forget_loader, split_user_train_dataset_to_remaining_forget
from utils import params_to_device
from model import (get_core_model_params, get_trained_linear, init_pretrained_model, split_model_to_feature_linear,
                   freeze, thaw)
from loss import MSELossDiv2

import torch
import random
import numpy as np

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'


def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


seed = 13  # any number
set_deterministic(seed=seed)

# loading core model and linearized model -- init in cpu
pretrained_model = init_pretrained_model('resnet50', 'cifar10')
_, linearized_head_core, __ = split_model_to_feature_linear(pretrained_model, 5, None,
                                                            send_params_to_device=False)
core_model_state_dict = get_core_model_params(
    'checkpoint/05042024-213334-train-user-data-resnet50-cifar10-last5'
    '/05042024_213334_train_user_data_resnet50_cifar10_last5_core_model.pth',
    'cpu')
feature_backbone, mixed_linear = get_trained_linear(
    'checkpoint/05042024-213334-train-user-data-resnet50-cifar10-last5'
    '/05042024_213334_train_user_data_resnet50_cifar10_last5.pth',
    'resnet50', 'cifar10', 5)
del _
del __

feature_backbone = feature_backbone.to(device)
freeze(feature_backbone)

mixed_linear = mixed_linear.to(device)
freeze(mixed_linear)

linearized_head_core = linearized_head_core.to(device)
freeze(linearized_head_core)

core_model_state_dict = params_to_device(core_model_state_dict, device)


def calculate_gradient(feature_backbone, core_model_state_dict, model, loss_fnc, regularizor_hyperparameter,
                       data_loader, device):
    grads = [torch.zeros_like(param) for param in model.parameters()]
    sample_count = 0
    thaw(model)
    for iter_idx, (inp, target) in enumerate(data_loader):
        model.zero_grad()
        inp = inp.to(device)
        target = 5 * target.to(device)
        curr_loss = loss_fnc(model(feature_backbone, core_model_state_dict, inp), target)
        curr_loss.backward()
        for idx, param in enumerate(model.parameters()):
            grads[idx] += (param.grad * inp.shape[0])
        sample_count += inp.shape[0]

        if iter_idx == 0 or (iter_idx + 1) % 50 == 0 or (iter_idx + 1) == len(data_loader):
            print('iter: {}/{}'.format(iter_idx + 1, len(data_loader)))
    freeze(model)

    last = []
    for grad, param in zip(grads, model.parameters()):
        tmp = grad / sample_count
        tmp = tmp + regularizor_hyperparameter * param.clone().detach()
        tmp.requires_grad = False
        last.append(tmp)
    return last


def calculate_hess_diag(feature_backbone, core_model_state_dict, model, loss_fnc, regularizor_hyperparameter,
                        data_loader, device):
    hess_diags = [torch.zeros_like(p) for p in model.parameters()]
    sample_count = 0
    v = [np.random.uniform(0, 1, size=p.shape) for p in model.parameters()]
    for vi in v:
        vi[vi < 0.5] = -1
        vi[vi >= 0.5] = 1
    v = [torch.tensor(vi) for vi in v]
    v = {key: param for (key, _), param in zip(model.named_parameters(), v)}
    v = params_to_device(v, device)

    thaw(model)
    for iter_idx, (inp, target) in enumerate(data_loader):
        model.zero_grad()
        inp = inp.to(device)
        target = 5 * target.to(device)
        curr_loss = loss_fnc(model(feature_backbone, core_model_state_dict, inp), target)
        curr_grad = torch.autograd.grad(curr_loss, model.parameters(), create_graph=True)

        vprod = None
        for vi, grad in zip(v.values(), curr_grad):
            if vprod is None:
                vprod = torch.sum(vi * grad)
            else:
                vprod += torch.sum(vi * grad)

        hvp_val = torch.autograd.grad(vprod, model.parameters())

        for idx, (vi, hvp_val_i) in enumerate(zip(v.values(), hvp_val)):
            hess_diags[idx] = hess_diags[idx] + (torch.abs(vi * hvp_val_i) * inp.shape[0])

        sample_count += inp.shape[0]

        if iter_idx == 0 or (iter_idx + 1) % 50 == 0 or (iter_idx + 1) == len(data_loader):
            print('iter: {}/{}'.format(iter_idx + 1, len(data_loader)))
    freeze(model)

    hess_diags = [(diags / sample_count) + (regularizor_hyperparameter * torch.norm(param) ** 2) for diags, param in
                  zip(hess_diags, v.values())]
    return hess_diags


def expected_hess_diag(feature_backbone, core_model_state_dict, model, loss_fnc, regularizor_hyperparameter,
                       data_loader, device, num_iter=20):
    expected_hess_diags = [torch.zeros_like(p) for p in model.parameters()]
    for iter in range(num_iter):
        print('#####expectation iter: {}#######\n'.format(iter + 1))
        hess_diags = calculate_hess_diag(feature_backbone, core_model_state_dict, model, loss_fnc,
                                         regularizor_hyperparameter, data_loader, device)
        for expected_idx in range(len(expected_hess_diags)):
            expected_hess_diags[expected_idx] = expected_hess_diags[expected_idx] + hess_diags[expected_idx]

    expected_hess_diags = [expected / num_iter for expected in expected_hess_diags]
    return expected_hess_diags


remain_dataset, _ = split_user_train_dataset_to_remaining_forget('cifar10', 'resnet50', 0.1, seed=13)
remain_loader, _ = get_remaining_forget_loader(remain_dataset, _, 64, shuffle=False)

remain_grads = calculate_gradient(feature_backbone, core_model_state_dict, mixed_linear, MSELossDiv2(), 0.0005,
                                  remain_loader, device)
remain_grads = [grad.to('cpu') for grad in remain_grads]

expected_hess_diags = expected_hess_diag(feature_backbone, core_model_state_dict, mixed_linear, MSELossDiv2(), 0.0005,
                                         remain_loader, device, num_iter=100)
feature_backbone = feature_backbone.to('cpu')
core_model_state_dict = params_to_device(core_model_state_dict, 'cpu')
remain_grads = [grad.to(device) for grad in remain_grads]
expected_hess_diags_inv = [1 / expected for expected in expected_hess_diags]
del expected_hess_diags
forgetting_update = [expected_inv * grad for expected_inv, grad in zip(expected_hess_diags_inv, remain_grads)]

forgetted_model = {name: first - second for (name, first), second in
                   zip(mixed_linear.tangents.items(), forgetting_update)}
with torch.no_grad():
    state_dict = mixed_linear.state_dict()
    for name, value in forgetted_model.items():
        state_dict['tangent_model.{}'.format(name)] = value
    mixed_linear.load_state_dict(state_dict)

torch.save({
    'model_state_dict': mixed_linear.state_dict(),
}, './hess_diag_model_100_iter.pth')
