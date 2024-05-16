import torch.nn as nn


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
