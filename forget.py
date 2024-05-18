import torch
import numpy as np
from utils import params_to_device
from model import thaw, freeze
import logging


def estimate_hess_inv_grad(feature_backbone, linearized_head_core, core_model_state_dict, v_param, optimizer,
                           jvp_norm_criterion, gradient_vector_inner_product_criterion, regularizor_criterion,
                           remain_loader, grads, device, weight_decay=0.0005):
    for epoch in range(150):
        if (epoch + 1) in [30, 60, 90, 120]:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5
        for iter_idx, (data, label) in enumerate(remain_loader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            jvp_norm_loss = 0.5 * jvp_norm_criterion(feature_backbone, linearized_head_core, core_model_state_dict,
                                                     v_param, data)
            gradient_vector_inner_product_loss = gradient_vector_inner_product_criterion(grads, v_param.values())
            regularizor_loss = 0.5 * weight_decay * regularizor_criterion(v_param.values())
            loss = jvp_norm_loss + regularizor_loss - gradient_vector_inner_product_loss
            loss.backward()
            optimizer.step()
            if iter_idx == 0 or (iter_idx + 1) % 50 == 0 or (iter_idx + 1) == len(remain_loader):
                print('epoch: {}/{}, iter: {}/{}, loss: {}'.format(epoch + 1, 150, iter_idx + 1, len(remain_loader),
                                                                   loss.item()))
                logging.info(
                    'epoch: {}/{}, iter: {}/{}, loss: {}'.format(epoch + 1, 150, iter_idx + 1, len(remain_loader),
                                                                 loss.item()))


def calculate_hess_diag(feature_backbone, core_model_state_dict, model, loss_fnc, regularizor_hyperparameter,
                        data_loader, device, activation_variant=False):
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
        if activation_variant:
            curr_loss = loss_fnc(model(core_model_state_dict, inp), target)
        else:
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

    hess_diags = [(diags / sample_count) + (regularizor_hyperparameter * (param * param)) for diags, param in
                  zip(hess_diags, v.values())]
    return hess_diags


def expected_hess_diag(feature_backbone, core_model_state_dict, model, loss_fnc, regularizor_hyperparameter,
                       data_loader, device, num_iter=20, activation_variant=False):
    expected_hess_diags = [torch.zeros_like(p) for p in model.parameters()]
    for iter in range(num_iter):
        print('#####expectation iter: {}#######\n'.format(iter + 1))
        hess_diags = calculate_hess_diag(feature_backbone, core_model_state_dict, model, loss_fnc,
                                         regularizor_hyperparameter, data_loader, device,
                                         activation_variant=activation_variant)
        for expected_idx in range(len(expected_hess_diags)):
            expected_hess_diags[expected_idx] = expected_hess_diags[expected_idx] + hess_diags[expected_idx]

    expected_hess_diags = [expected / num_iter for expected in expected_hess_diags]
    return expected_hess_diags

