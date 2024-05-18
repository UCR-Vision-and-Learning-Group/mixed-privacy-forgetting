import torch
from torch.optim import SGD, lr_scheduler, Adam

import argparse
import numpy as np
import random

from utils import *
from train import *
from loss import *
from dataset import *
from model import *
from forget import *


def set_deterministic_environment(seed=13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def train_user_data(arch_id, dataset_id, number_of_linearized_components,
                    use_default=True, pretrained_model_path=None,
                    device_id=0, shuffle=True, split_rate=0, weight_decay=0.0005,
                    init_hidden_layers=None, activation_variant=False):
    name_arr = [arch_id, dataset_id, 'last{}'.format(number_of_linearized_components)]
    if split_rate > 0:
        name_arr = name_arr + ['split{}'.format(split_rate)]

    exp_path = init_exp('train-user-data', name_arr)

    device = 'cuda:{}'.format(device_id) if torch.cuda.is_available() else 'cpu'
    pretrained_model = init_pretrained_model(arch_id, dataset_id.split('-')[0], use_default=use_default,
                                             pretrained_model_path=pretrained_model_path,
                                             hidden_layers=init_hidden_layers)

    split_before = init_hidden_layers is not None
    feature_model, linear_model, linear_model_params = split_model_to_feature_linear(pretrained_model,
                                                                                     number_of_linearized_components,
                                                                                     device, split_before=split_before)
    torch.save({
        'params': linear_model_params
    }, get_core_model_path(exp_path))

    if activation_variant:
        feature_model = feature_model.to('cpu')
        del feature_model
        feature_model = None

    if activation_variant:
        mixed_linear_model = MixedLinearActivationVariant(linear_model)
    else:
        mixed_linear_model = MixedLinear(linear_model)
    mixed_linear_model = mixed_linear_model.to(device)

    criterion = LossWrapper([MSELossDiv2(), L2Regularization()], [1, weight_decay])
    optimizer = SGD(mixed_linear_model.parameters(), lr=0.05, momentum=0.9)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[24, 39], gamma=0.1)

    if pretrained_model_path is not None:
        core_dataset, user_train_dataset, user_test_dataset = split_dataset_to_core_user(dataset_id, arch_id,
                                                                                         split_rate, seed=13,
                                                                                         number_of_linearized_components=number_of_linearized_components)
        _, user_train_loader, user_test_loader = get_core_user_loader(core_dataset, user_train_dataset,
                                                                      user_test_dataset, 64, shuffle=shuffle)
    elif split_rate > 0:
        remaining_dataset, _ = split_user_train_dataset_to_remaining_forget(dataset_id, arch_id, split_rate, seed=13,
                                                                            number_of_linearized_components=number_of_linearized_components)
        user_train_loader, _ = get_remaining_forget_loader(remaining_dataset, _, 64, shuffle=True)
        _, user_test_loader = get_user_loader(dataset_id, arch_id, 64, shuffle=shuffle,
                                              number_of_linearized_components=number_of_linearized_components)
    else:
        user_train_loader, user_test_loader = get_user_loader(dataset_id, arch_id, 64, shuffle=shuffle,
                                                              number_of_linearized_components=number_of_linearized_components)

    running_loss = []
    running_test_acc = []
    running_train_acc = []

    best_model_test_acc = -1
    best_model_epoch = -1

    init_checkpoint(running_loss, running_test_acc, running_train_acc, best_model_test_acc, best_model_epoch, exp_path)

    epoch = None
    for epoch in range(50):
        checkpoint = get_checkpoint(exp_path)
        running_test_acc, checkpoint = test_mixed_linear(mixed_linear_model, user_test_loader, feature_model,
                                                         linear_model_params, optimizer, running_test_acc, epoch,
                                                         device,
                                                         checkpoint, best_model_test_acc, best_model_epoch,
                                                         activation_variant=activation_variant)
        running_train_acc, checkpoint = train_accuracy_mixed_linear(mixed_linear_model, user_train_loader,
                                                                    feature_model,
                                                                    linear_model_params, running_train_acc, epoch,
                                                                    device,
                                                                    checkpoint,
                                                                    activation_variant=activation_variant)
        mixed_linear_model, optimizer, scheduler, running_loss, checkpoint = train_mixed_linear(mixed_linear_model,
                                                                                                user_train_loader,
                                                                                                feature_model,
                                                                                                linear_model_params,
                                                                                                optimizer, criterion,
                                                                                                scheduler,
                                                                                                running_loss, device,
                                                                                                epoch, checkpoint,
                                                                                                activation_variant=activation_variant)
        set_checkpoint(checkpoint, exp_path)

    checkpoint = get_checkpoint(exp_path)
    running_test_acc, checkpoint = test_mixed_linear(mixed_linear_model, user_test_loader, feature_model,
                                                     linear_model_params, optimizer, running_test_acc, epoch, device,
                                                     checkpoint, best_model_test_acc, best_model_epoch,
                                                     activation_variant=activation_variant)
    running_train_acc, checkpoint = train_accuracy_mixed_linear(mixed_linear_model, user_train_loader, feature_model,
                                                                linear_model_params, running_train_acc, epoch, device,
                                                                checkpoint, activation_variant=activation_variant)
    set_checkpoint(checkpoint, exp_path)


def pretrain(arch_id, dataset_id, split_rate, device_id=0, shuffle=True, init_hidden_layers=None):
    exp_path = init_exp('pretrain', [arch_id, dataset_id, 'split{}'.format(split_rate)])

    device = 'cuda:{}'.format(device_id) if torch.cuda.is_available() else 'cpu'
    pretrained_model = init_pretrained_model(arch_id, dataset_id, use_default=True, pretrained_model_path=None,
                                             hidden_layers=init_hidden_layers)
    pretrained_model = pretrained_model.to(device)

    core_dataset, user_train_dataset, user_test_dataset = split_dataset_to_core_user(dataset_id, arch_id, split_rate,
                                                                                     seed=13)
    core_train_loader, _, user_test_loader = get_core_user_loader(core_dataset, user_train_dataset, user_test_dataset,
                                                                  64, shuffle=shuffle)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(pretrained_model.parameters(), lr=0.001)

    running_loss = []
    running_test_acc = []
    running_train_acc = []

    best_model_test_acc = -1
    best_model_epoch = -1

    init_checkpoint(running_loss, running_test_acc, running_train_acc, best_model_test_acc, best_model_epoch, exp_path)

    epoch = None
    for epoch in range(20):
        checkpoint = get_checkpoint(exp_path)
        running_test_acc, checkpoint = test_pretrain(pretrained_model, user_test_loader, optimizer,
                                                     running_test_acc, epoch, device, checkpoint,
                                                     best_model_test_acc, best_model_epoch)

        pretrained_model, optimizer, running_loss, checkpoint = train_pretrain(pretrained_model, core_train_loader,
                                                                               optimizer, criterion, running_loss,
                                                                               device, epoch, checkpoint)
        set_checkpoint(checkpoint, exp_path)

    checkpoint = get_checkpoint(exp_path)
    running_test_acc, checkpoint = test_pretrain(pretrained_model, user_test_loader, optimizer,
                                                 running_test_acc, epoch, device, checkpoint,
                                                 best_model_test_acc, best_model_epoch)

    set_checkpoint(checkpoint, exp_path)


def save_activations(arch_id, dataset_id, number_of_linearized_components, device_id=0):
    name_arr = [arch_id, dataset_id, 'last{}'.format(number_of_linearized_components)]
    data_root_path = os.path.join('./data', '-'.join(name_arr))
    if not os.path.exists(data_root_path):
        os.makedirs(data_root_path)

    device = 'cuda:{}'.format(device_id) if torch.cuda.is_available() else 'cpu'
    pretrained_model = init_pretrained_model(arch_id, dataset_id, use_default=True, pretrained_model_path=None,
                                             hidden_layers=None)
    # TODO: hidden layers property can be added
    feature_model, _, __ = split_model_to_feature_linear(pretrained_model,
                                                         number_of_linearized_components,
                                                         device, split_before=False)

    train_activations = None
    train_labels = None
    test_activations = None
    test_labels = None
    user_train_loader, user_test_loader = get_user_loader(dataset_id, arch_id, 256, shuffle=False)
    with torch.no_grad():
        # TODO: this loop would be a function
        for iter, (data, label) in enumerate(user_train_loader):
            data = data.to(device)
            activations = feature_model(data)
            if train_activations is None:
                train_activations = activations.clone().detach().to('cpu')
            else:
                train_activations = torch.cat([train_activations, activations.clone().detach().to('cpu')])

            if train_labels is None:
                train_labels = torch.argmax(label, dim=1)
            else:
                train_labels = torch.cat([train_labels, torch.argmax(label, dim=1)])

            if (iter + 1) % 20 == 0 or (iter + 1) == len(user_train_loader):
                print('{}/{}'.format(iter + 1, len(user_train_loader)))

        torch.save(
            {
                'data': train_activations,
                'label': train_labels
            }, os.path.join(data_root_path, 'train_data.pth')
        )

        # noinspection PyShadowingBuiltins
        for iter, (data, label) in enumerate(user_test_loader):
            data = data.to(device)
            activations = feature_model(data)
            if test_activations is None:
                test_activations = activations.clone().detach().to('cpu')
            else:
                test_activations = torch.cat([test_activations, activations.clone().detach().to('cpu')])

            if test_labels is None:
                test_labels = label
            else:
                test_labels = torch.cat([test_labels, label])

            if (iter + 1) % 20 == 0 or (iter + 1) == len(user_test_loader):
                print('{}/{}'.format(iter + 1, len(user_test_loader)))

        torch.save(
            {
                'data': test_activations,
                'label': test_labels
            }, os.path.join(data_root_path, 'test_data.pth')
        )


def mixed_privacy(arch_id, dataset_id, number_of_linearized_components, split_rate, checkpoint_path,
                  device_id=0, weight_decay=0.0005, activation_variant=False):
    name_arr = [arch_id, dataset_id, 'last{}'.format(number_of_linearized_components), 'split{}'.format(split_rate)]
    exp_path = init_exp('mixed-privacy', name_arr)

    # loading core model and linearized model -- init in cpu
    pretrained_model = init_pretrained_model(arch_id, dataset_id.split('-')[0])
    _, linearized_head_core, __ = split_model_to_feature_linear(pretrained_model,
                                                                number_of_linearized_components,
                                                                None, send_params_to_device=False)
    path_base_name = '_'.join(os.path.split(checkpoint_path)[1].split('-'))

    core_model_state_dict = get_core_model_params(os.path.join(checkpoint_path,
                                                               '{}_core_model.pth'.format(path_base_name)),
                                                  'cpu')
    feature_backbone, mixed_linear = get_trained_linear(os.path.join(checkpoint_path, '{}.pth'.format(path_base_name)),
                                                        arch_id, dataset_id.split('-')[0],
                                                        number_of_linearized_components,
                                                        activation_variant=activation_variant)
    del _
    del __

    if activation_variant:
        feature_backbone = feature_backbone.to('cpu')
        del feature_backbone
        feature_backbone = None

    v_param = {key: torch.randn_like(value, device='cpu') for key, value in
               core_model_state_dict.items()}  # init in cpu

    device = 'cuda:{}'.format(device_id) if torch.cuda.is_available() else 'cpu'

    if not activation_variant:
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

    remaining_dataset, forget_dataset = split_user_train_dataset_to_remaining_forget(dataset_id, arch_id, split_rate,
                                                                                     number_of_linearized_components=number_of_linearized_components)
    remain_loader, forget_loader = get_remaining_forget_loader(remaining_dataset, forget_dataset, 256)

    main_criterion = LossWrapper([MSELossDiv2(), L2Regularization()], [1, weight_decay])
    grads = calculate_gradient(feature_backbone, core_model_state_dict, mixed_linear, main_criterion, remain_loader,
                               device, activation_variant=activation_variant)

    grad_path = os.path.split(exp_path)[0]
    grad_path = os.path.join(grad_path, '{}_grads.pth'.format('_'.join(os.path.split(grad_path)[1].split('-'))))
    torch.save({
        'grads': grads,
    }, grad_path)

    jvp_norm_criterion = JVPNormLoss(activation_variant=activation_variant)
    gradient_vector_inner_product_criterion = GradientVectorInnerProduct()
    regularizor_criterion = L2Regularization()

    optimizer = SGD(v_param.values(), lr=0.001, momentum=0.999)

    estimate_hess_inv_grad(feature_backbone, linearized_head_core, core_model_state_dict, v_param, optimizer,
                           jvp_norm_criterion, gradient_vector_inner_product_criterion, regularizor_criterion,
                           remain_loader, grads, device, activation_variant=activation_variant,
                           weight_decay=weight_decay)

    v_param_path = os.path.split(exp_path)[0]
    v_param_path = os.path.join(v_param_path,
                                '{}_v_param.pth'.format('_'.join(os.path.split(v_param_path)[1].split('-'))))
    torch.save({
        'v_param': v_param,
    }, v_param_path)

    forgetted = {name: first - second for name, first, second in
                 zip(mixed_linear.tangents.keys(), mixed_linear.tangents.values(), v_param.values())}

    with torch.no_grad():
        state_dict = mixed_linear.state_dict()
        for name, value in forgetted.items():
            state_dict['tangent_model.{}'.format(name)] = value
        mixed_linear.load_state_dict(state_dict)

    torch.save({
        'model_state_dict': mixed_linear.state_dict(),
    }, exp_path)


def forget_by_diag(arch_id, dataset_id, number_of_linearized_components, split_rate, checkpoint_path,
                   device_id=0, weight_decay=0.0005, activation_variant=False, num_iter=100):
    name_arr = [arch_id, dataset_id, 'last{}'.format(number_of_linearized_components), 'split{}'.format(split_rate),
                'iter{}'.format(num_iter)]
    exp_path = init_exp('forget-by-diag', name_arr)

    # loading core model and linearized model -- init in cpu
    path_base_name = '_'.join(os.path.split(checkpoint_path)[1].split('-'))

    core_model_state_dict = get_core_model_params(os.path.join(checkpoint_path,
                                                               '{}_core_model.pth'.format(path_base_name)),
                                                  'cpu')
    feature_backbone, mixed_linear = get_trained_linear(os.path.join(checkpoint_path, '{}.pth'.format(path_base_name)),
                                                        arch_id, dataset_id.split('-')[0],
                                                        number_of_linearized_components,
                                                        activation_variant=activation_variant)

    if activation_variant:
        feature_backbone = feature_backbone.to('cpu')
        del feature_backbone
        feature_backbone = None

    device = 'cuda:{}'.format(device_id) if torch.cuda.is_available() else 'cpu'

    if not activation_variant:
        feature_backbone = feature_backbone.to(device)
        freeze(feature_backbone)

    mixed_linear = mixed_linear.to(device)
    freeze(mixed_linear)

    core_model_state_dict = params_to_device(core_model_state_dict, device)

    remain_dataset, _ = split_user_train_dataset_to_remaining_forget(dataset_id, arch_id, split_rate,
                                                                     number_of_linearized_components=number_of_linearized_components)
    remain_loader, _ = get_remaining_forget_loader(remain_dataset, _, 64, shuffle=False)

    main_criterion = LossWrapper([MSELossDiv2(), L2Regularization()], [1, weight_decay])

    remain_grads = calculate_gradient(feature_backbone, core_model_state_dict, mixed_linear, main_criterion,
                                      remain_loader, device, activation_variant=activation_variant)

    remain_grads = [grad.to('cpu') for grad in remain_grads]

    expected_hess_diags = expected_hess_diag(feature_backbone, core_model_state_dict, mixed_linear, MSELossDiv2(),
                                             weight_decay, remain_loader, device, num_iter=num_iter,
                                             activation_variant=activation_variant)

    if feature_backbone is not None:
        feature_backbone = feature_backbone.to('cpu')
        del feature_backbone
        feature_backbone = None

    core_model_state_dict = params_to_device(core_model_state_dict, 'cpu')
    del core_model_state_dict
    core_model_state_dict = None

    remain_grads = [grad.to(device) for grad in remain_grads]
    remain_grads_path = os.path.split(exp_path)[0]
    remain_grads_path = os.path.join(remain_grads_path,
                                     '{}_remaining_grads.pth'.format(
                                         '_'.join(os.path.split(remain_grads_path)[1].split('-'))))
    torch.save({'remaining_grads': remain_grads}, remain_grads_path)

    expected_hess_diags_inv = [1 / expected for expected in expected_hess_diags]

    expected_hess_diags_path = os.path.split(exp_path)[0]
    expected_hess_diags_path = os.path.join(expected_hess_diags_path,
                                            '{}_expected_hess_diags.pth'.format(
                                                '_'.join(os.path.split(expected_hess_diags_path)[1].split('-'))))
    torch.save({'expected_hess_diags': expected_hess_diags}, expected_hess_diags_path)

    expected_hess_diags_inv_path = os.path.split(exp_path)[0]
    expected_hess_diags_inv_path = os.path.join(expected_hess_diags_inv_path,
                                                '{}_expected_hess_diags_inv.pth'.format(
                                                    '_'.join(
                                                        os.path.split(expected_hess_diags_inv_path)[1].split('-'))))
    torch.save({'expected_hess_diags_inv': expected_hess_diags_inv}, expected_hess_diags_inv_path)

    del expected_hess_diags
    forgetting_update = [expected_inv * grad for expected_inv, grad in zip(expected_hess_diags_inv, remain_grads)]

    forgetting_update_path = os.path.split(exp_path)[0]
    forgetting_update_path = os.path.join(forgetting_update_path,
                                          '{}_forgetting_update.pth'.format(
                                              '_'.join(
                                                  os.path.split(forgetting_update_path)[1].split('-'))))
    torch.save({'forgetting_update': forgetting_update}, forgetting_update_path)

    forgetted_model = {name: first - second for (name, first), second in
                       zip(mixed_linear.tangents.items(), forgetting_update)}
    with torch.no_grad():
        state_dict = mixed_linear.state_dict()
        for name, value in forgetted_model.items():
            state_dict['tangent_model.{}'.format(name)] = value
        mixed_linear.load_state_dict(state_dict)

    torch.save({
        'model_state_dict': mixed_linear.state_dict(),
    }, exp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--mode', dest='mode', type=str, required=True)

    parser.add_argument('-di', '--dataset-id', dest='dataset_id', type=str, required=True)
    parser.add_argument('-ai', '--arch-id', dest='arch_id', type=str, required=True)
    parser.add_argument('-nlc', '--number-of-linearized-components', dest='number_of_linearized_components', type=int)

    parser.add_argument('-dei', '--device-id', dest='device_id', type=int, default=0)
    parser.add_argument('-ud', '--use-default', dest='use_default', action='store_true')
    parser.add_argument('-pmp', '--pretrained-model-path', dest='pretrained_model_path', type=str)
    parser.add_argument('-cp', '--checkpoint-path', dest='checkpoint_path', type=str)
    parser.add_argument('-sr', '--split-rate', dest='split_rate', type=float, default=0)  # TODO: handle the exceptions

    parser.add_argument('-wd', '--weight-decay', dest='weight_decay', type=float, default=0.0005)
    parser.add_argument('-ihd', '--init-hidden-layers', dest='init_hidden_layers', nargs='*', type=int)
    parser.add_argument('-av', '--activation-variant', dest='activation_variant', action='store_true')
    parser.add_argument('-ni', '--num-iter-for-diag', dest='num_iter', type=int, default=100)

    args = parser.parse_args()

    set_deterministic_environment()

    if args.mode == 'train-user-data':
        train_user_data(args.arch_id, args.dataset_id, args.number_of_linearized_components,
                        use_default=args.use_default, pretrained_model_path=args.pretrained_model_path,
                        device_id=args.device_id, split_rate=args.split_rate,
                        init_hidden_layers=args.init_hidden_layers, activation_variant=args.activation_variant)
    elif args.mode == 'pretrain':
        pretrain(args.arch_id, args.dataset_id, args.split_rate, device_id=args.device_id, shuffle=True,
                 init_hidden_layers=list(args.init_hidden_layers))
    elif args.mode == 'save-activations':
        save_activations(args.arch_id, args.dataset_id, args.number_of_linearized_components, device_id=args.device_id)
    elif args.mode == 'mixed-privacy':
        mixed_privacy(args.arch_id, args.dataset_id, args.number_of_linearized_components, args.split_rate,
                      args.checkpoint_path, device_id=args.device_id, weight_decay=args.weight_decay,
                      activation_variant=args.activation_variant)
    elif args.mode == 'forget-by-diag':
        forget_by_diag(args.arch_id, args.dataset_id, args.number_of_linearized_components, args.split_rate,
                       args.checkpoint_path, device_id=args.device_id, weight_decay=args.weight_decay,
                       activation_variant=args.activation_variant, num_iter=args.num_iter)
