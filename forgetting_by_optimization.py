# libraries
import torch
from model import get_core_model_params, get_trained_linear, freeze
from dataset import split_user_train_dataset_to_remaining_forget, get_remaining_forget_loader
from utils import params_to_device
from loss import MSELossDiv2
import os
import cvxpy as cp
import numpy as np

device = 'cpu:0' if torch.cuda.is_available() else 'cpu'

# load pretrained model
exp_path = 'checkpoint/05152024-011132-train-user-data-resnet18-cifar10-last1/'
core_model_state_dict = get_core_model_params(os.path.join(exp_path, '05152024_011132_train_user_data_resnet18_cifar10_last1_core_model.pth'), 'cpu')
_, mixed_linear = get_trained_linear(os.path.join(exp_path, '05152024_011132_train_user_data_resnet18_cifar10_last1.pth'), 'resnet18', 'cifar10', 1, activation_variant=True)
del _
exp_path = './checkpoint/tmp/'

mixed_linear = mixed_linear.to(device)
freeze(mixed_linear)

core_model_state_dict = params_to_device(core_model_state_dict, device)

## split dataset into remaning and forget
remaining_dataset, forget_dataset = split_user_train_dataset_to_remaining_forget('cifar10-act', 'resnet18', 0.1, number_of_linearized_components=1)
remain_loader, forget_loader = get_remaining_forget_loader(remaining_dataset, forget_dataset, 256)

# calculate the hessian on the last linear layer for both remaning and forget
def calculate_hessian(loader, exp_path, mode='forget'):
    print('{} hessian'.format(mode))
    hessian = None
    sample_count = 0
    with torch.no_grad():
        for iter, (data, _) in enumerate(loader):
            data = data.to(device)
            act = data.unsqueeze(-1)
            batched_hessian = act @ act.permute(0, 2, 1)
            if hessian is None:
                hessian = torch.sum(batched_hessian, dim=0).clone().detach().to('cpu')
            else:
                hessian += torch.sum(batched_hessian, dim=0).clone().detach().to('cpu')
            sample_count += data.shape[0]
            if (iter + 1) % 50 == 0 or (iter + 1) == len(loader):
                print('iter: {}/{}'.format(iter + 1, len(loader))) 
    hessian = hessian / sample_count + 0.005 * torch.eye(hessian.shape[0])
    torch.save({'hessian': hessian}, os.path.join(exp_path, '05152024_011132_train_user_data_resnet18_cifar10_last1_{}_hessian.pth'.format(mode)))
    return hessian

forget_hessian = calculate_hessian(forget_loader, exp_path, mode='forget')
remain_hessian = calculate_hessian(remain_loader, exp_path, mode='remain')

# sample perturbed parameters
## perturb from gradient direction
## NOTE: we can analyze the effects of sampling different perturbations and its importance

trained_mixed_linear_weights = [key.clone().detach().to('cpu') for key in mixed_linear.tangents.values()]
num_of_perturbations = 500
scale_random = 0.01

# using default random perturbation
perturbations = []
perturbed_weights = []
for _ in range(num_of_perturbations):
    curr_perturb = [torch.randn(*weight.shape) * scale_random for weight in trained_mixed_linear_weights]
    curr_perturbed_weight = [weight + perturb for weight, perturb in zip(trained_mixed_linear_weights, curr_perturb)]
    perturbations.append(curr_perturb)
    perturbed_weights.append(curr_perturbed_weight)
torch.save({'perturbations': perturbations}, os.path.join(exp_path, '05152024_011132_train_user_data_resnet18_cifar10_last1_perturbations.pth'))
torch.save({'perturbed_weights': perturbed_weights}, os.path.join(exp_path, '05152024_011132_train_user_data_resnet18_cifar10_last1_perturbed_weights.pth'))

# find out loss differences (L_forget)
criterion = MSELossDiv2()
forget_loss_differences = torch.zeros(num_of_perturbations).to(device)
sample_count = 0
mixed_linear.eval()
with torch.no_grad():
    for iter, (data, label) in enumerate(forget_loader):
        data, label = data.to(device), label.to(device)
        label = label * 5
        preds = mixed_linear(core_model_state_dict, data)
        actual_loss = criterion(preds, label)
        sample_count += data.shape[0]
        for perturb_idx, perturbed_weight in enumerate(perturbed_weights):
            mixed_linear.to('cpu')
            state_dict = mixed_linear.state_dict()
            for key, perturbed in zip(state_dict.keys(), perturbed_weight):
                state_dict[key] = perturbed
            mixed_linear.load_state_dict(state_dict)
            mixed_linear.to(device)
            perturbed_preds = mixed_linear(core_model_state_dict, data)
            perturbed_loss = criterion(perturbed_preds, label)
            forget_loss_differences[perturb_idx] += (perturbed_loss - actual_loss) * 2 * data.shape[0]
            if (perturb_idx + 1) % 10 == 0 or (perturb_idx + 1) == len(forget_loader):
                print('iter: {}/{} perturb: {}/{}'.format(iter + 1, len(forget_loader), perturb_idx + 1, len(perturbed_weights)))
    forget_loss_differences = forget_loss_differences / sample_count
    forget_loss_differences = forget_loss_differences.to('cpu')

exact_hessian = (forget_hessian * len(forget_dataset) + remain_hessian * len(remaining_dataset)) / (len(remaining_dataset) + len(forget_dataset))

# stacked_linearized_perturbations = np.empty((perturbations[0][0].shape[0] * perturbations[0][0].shape[1], num_of_perturbations))
# for perturbation_idx, perturbation in enumerate(perturbations):
#     stacked_linearized_perturbations[:, perturbation_idx] = perturbation[0].numpy()

# set optimization problem
H = cp.Variable(exact_hessian.shape)
# exact_H = cp.kron(H, np.eye(perturbations[0][0].shape[0]))
# losses = cp.matmul(stacked_linearized_perturbations.T, cp.matmul(exact_H, stacked_linearized_perturbations))
loss = None
for perturbation in perturbations:
    perturbation = perturbation[0].numpy()
    if loss is None:
        loss = cp.trace(cp.matmul(perturbation, cp.matmul(H, perturbation.T)))
    else:
        loss = loss + cp.trace(cp.matmul(perturbation, cp.matmul(H, perturbation.T)))
loss = (loss / (2 * len(perturbations))) - np.mean(forget_loss_differences.numpy())
# loss = (cp.trace(losses) / (2 * losses.shape[0])) - np.mean(forget_loss_differences.numpy())
objective = cp.Minimize(loss)
constraints = [H >> 0, cp.trace(H) >= 0, H >> forget_hessian.numpy()]
problem = cp.Problem(objective, constraints)
problem.solve(verbose=True)

H = torch.tensor(H.value)
torch.save({
    'predicted_hessian': H  
}, os.path.join(exp_path, '05152024_011132_train_user_data_resnet18_cifar10_last1_predicted_hessian.pth'))