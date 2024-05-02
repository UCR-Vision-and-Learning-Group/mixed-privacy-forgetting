from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
from torch.func import functional_call
import torch.autograd.forward_ad as fwAD
import torch
from torch.optim import SGD, lr_scheduler
from dataset import get_train_loader
import logging
from datetime import datetime
import os
import sys


curr_file_name = sys.argv[0].split('_')
curr_file_name[-1] = curr_file_name[-1].split('.')[0]
now = datetime.now()
folder_name_datetime = now.strftime('%m%d%Y-%H%M%S')
file_name_datetime = now.strftime('%m%d%Y_%H%M%S')
folder_name_exp = '-'.join(curr_file_name)
file_name_exp = '_'.join(curr_file_name)
folder_name_exp = '{}-{}'.format(folder_name_datetime, folder_name_exp)
file_name_exp = '{}_{}'.format(file_name_datetime, file_name_exp)

exp_dir = os.path.join('./checkpoint/', folder_name_exp)
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

log_path = os.path.join(exp_dir, '{}.log'.format(file_name_exp))
logging.basicConfig(filename=log_path, level=logging.INFO)

exp_path = os.path.join(exp_dir, '{}.pth'.format(file_name_exp))

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
    
model = resnet50(weights=ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# splitting model
kid_arr = []
for kid in model.children():
    grand_kid_arr = [c for c in kid.children()]
    if len(grand_kid_arr) > 0:
        for grand_kid in grand_kid_arr:
            kid_arr.append(grand_kid)
    else:
        kid_arr.append(kid)

feature_backbone = nn.Sequential(*kid_arr[:-5])
freeze(feature_backbone)
feature_backbone = feature_backbone.to(device)


linearized_head_core = kid_arr[-5:]
linearized_head_core.insert(len(linearized_head_core) - 1, Flatten())

linearized_head_core = nn.Sequential(*linearized_head_core)
params = {name: p.detach().clone().to(device) for name, p in linearized_head_core.named_parameters()}

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
    
mixed_linear = MixedLinear(linearized_head_core)
mixed_linear = mixed_linear.to(device)

class L2Regularization(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.need_params = True

    def forward(self, params):
        l2_loss = None
        for param in params:       
            if l2_loss is None:
                l2_loss = param.norm(2)
            else:
                l2_loss = l2_loss + param.norm(2)
        return l2_loss

class LossWrapper(nn.Module):
    def __init__(self, loss_modules, hyperparameters):
        super().__init__()
        self.loss_modules = loss_modules
        self.hyperparameters = hyperparameters
    
    def forward(self, input, target, params):
        loss = None
        for loss_module, hyperparameter in zip(self.loss_modules, self.hyperparameters):
            if loss is None:
                if hasattr(loss_module, 'need_params'):
                    loss = loss_module(params) * hyperparameter
                else:
                    loss = loss_module(input, target) * hyperparameter
            else:
                if hasattr(loss_module, 'need_params'):
                    loss += loss_module(params) * hyperparameter
                else:
                    loss += loss_module(input, target) * hyperparameter

        return loss
    
criterion = LossWrapper([nn.MSELoss(), L2Regularization()], [1, 0.00001])
optimizer = SGD(mixed_linear.parameters(), lr=0.05, momentum=0.9)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[24, 39], gamma=0.1)

train_loader, test_loader = get_train_loader('cifar10', 'resnet18', 64)

running_loss = []
running_test_acc = []
running_train_acc = []

best_model_test_acc = -1
best_model_epoch = -1

torch.save({
    'running_loss': running_loss,
    'running_test_acc': running_test_acc,
    'running_train_acc': running_train_acc,
    'best_model_test_acc': best_model_test_acc,
    'best_model_epoch': best_model_epoch,
    'model_state_dict': None,
    'optimizer_state_dict': None 
}, exp_path)

for epoch in range(50):
    checkpoint = torch.load(exp_path)
    mixed_linear.eval()
    with torch.no_grad():
        true_count = 0
        sample_count = 0
        for test_iter_idx, (test_data, test_label) in enumerate(test_loader):
            test_data, test_label = test_data.to(device), test_label.to(device)
            preds = mixed_linear(feature_backbone, params, test_data)
            predicted_label = torch.argmax(preds, dim=1)
            true_count += torch.count_nonzero(predicted_label == test_label).item()
            sample_count += test_data.shape[0]
            if test_iter_idx == 0 or (test_iter_idx + 1) % 25 == 0 or (test_iter_idx + 1) == len(test_loader):
                print('test iter - processing: {}/{}'.format(test_iter_idx + 1, len(test_loader)))
        print('epoch: {}/{}, test accuracy: {}'.format(epoch + 1, 50, true_count / sample_count))
        logging.info('epoch: {}/{}, test accuracy: {}'.format(epoch + 1, 50, true_count / sample_count))
        running_test_acc.append(true_count / sample_count)
        checkpoint['running_test_acc'] = running_test_acc
        
        if (true_count / sample_count) > best_model_test_acc:
            best_model_test_acc = (true_count / sample_count)
            best_model_epoch = epoch + 1
            checkpoint['best_model_test_acc'] = best_model_test_acc
            checkpoint['best_model_epoch'] = best_model_epoch
            checkpoint['model_state_dict'] = mixed_linear.state_dict()
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        true_count = 0
        sample_count = 0
        for train_iter_idx, (train_data, train_label) in enumerate(train_loader):
            train_data, train_label = train_data.to(device), train_label.to(device)
            preds = mixed_linear(feature_backbone, params, train_data)
            predicted_label = torch.argmax(preds, dim=1)
            ground_truth_label = torch.argmax(train_label, dim=1)
            true_count += torch.count_nonzero(predicted_label == ground_truth_label).item()
            sample_count += train_data.shape[0]
            if train_iter_idx == 0 or (train_iter_idx + 1) % 100 == 0 or (train_iter_idx + 1) == len(train_loader):
                print('train iter - processing: {}/{}'.format(train_iter_idx + 1, len(train_loader)))
        print('epoch: {}/{}, train accuracy: {}'.format(epoch + 1, 50, true_count / sample_count))
        logging.info('epoch: {}/{}, train accuracy: {}'.format(epoch + 1, 50, true_count / sample_count))
        running_train_acc.append(true_count / sample_count)
        checkpoint['running_train_acc'] = running_train_acc

    mixed_linear.train()
    for iter_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        label = label * 5
        optimizer.zero_grad()
        preds = mixed_linear(feature_backbone, params, data)
        loss = criterion(preds, label, mixed_linear.parameters())
        loss.backward()
        optimizer.step()
        curr_lr = optimizer.param_groups[0]["lr"]
        running_loss.append(loss.item())
        if iter_idx == 0 or (iter_idx + 1) % 100 == 0 or (iter_idx + 1) == len(train_loader):
            print('epoch: {}/{}, iter: {}/{}, lr: {}, loss: {}'.format(epoch + 1, 50, iter_idx + 1, len(train_loader), curr_lr, loss.item()))
            logging.info('epoch: {}/{}, iter: {}/{}, lr: {}, loss: {}'.format(epoch + 1, 50, iter_idx + 1, len(train_loader), curr_lr, loss.item()))
    scheduler.step()
    checkpoint['running_loss'] = running_loss
    torch.save(checkpoint, exp_path)

checkpoint = torch.load(exp_path)
mixed_linear.eval()
with torch.no_grad():
    true_count = 0
    sample_count = 0
    for test_iter_idx, (test_data, test_label) in enumerate(test_loader):
        test_data, test_label = test_data.to(device), test_label.to(device)
        preds = mixed_linear(feature_backbone, params, test_data)
        predicted_label = torch.argmax(preds, dim=1)
        true_count += torch.count_nonzero(predicted_label == test_label).item()
        sample_count += test_data.shape[0]
        if test_iter_idx == 0 or (test_iter_idx + 1) % 25 == 0 or (test_iter_idx + 1) == len(test_loader):
            print('test iter - processing: {}/{}'.format(test_iter_idx + 1, len(test_loader)))
    print('epoch: {}/{}, test accuracy: {}'.format(epoch + 1, 50, true_count / sample_count))
    logging.info('epoch: {}/{}, test accuracy: {}'.format(epoch + 1, 50, true_count / sample_count))
    running_test_acc.append(true_count / sample_count)
    checkpoint['running_test_acc'] = running_test_acc
    
    if (true_count / sample_count) > best_model_test_acc:
            best_model_test_acc = (true_count / sample_count)
            best_model_epoch = epoch + 1
            checkpoint['best_model_test_acc'] = best_model_test_acc
            checkpoint['best_model_epoch'] = best_model_epoch
            checkpoint['model_state_dict'] = mixed_linear.state_dict()
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    true_count = 0
    sample_count = 0
    for train_iter_idx, (train_data, train_label) in enumerate(train_loader):
        train_data, train_label = train_data.to(device), train_label.to(device)
        preds = mixed_linear(feature_backbone, params, train_data)
        predicted_label = torch.argmax(preds, dim=1)
        ground_truth_label = torch.argmax(train_label, dim=1)
        true_count += torch.count_nonzero(predicted_label == ground_truth_label).item()
        sample_count += train_data.shape[0]
        if train_iter_idx == 0 or (train_iter_idx + 1) % 100 == 0 or (train_iter_idx + 1) == len(train_loader):
            print('train iter - processing: {}/{}'.format(train_iter_idx + 1, len(train_loader)))
    print('epoch: {}/{}, train accuracy: {}'.format(epoch + 1, 50, true_count / sample_count))
    logging.info('epoch: {}/{}, train accuracy: {}'.format(epoch + 1, 50, true_count / sample_count))
    running_train_acc.append(true_count / sample_count)
    checkpoint['running_train_acc'] = running_train_acc
torch.save(checkpoint, exp_path)