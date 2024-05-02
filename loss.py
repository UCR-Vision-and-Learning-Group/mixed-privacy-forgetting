import torch.nn as nn

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