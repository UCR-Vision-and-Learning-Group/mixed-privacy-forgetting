import torch
import torch.nn as nn
from torch.optim import Adam

from torchvision.models import resnet18, ResNet18_Weights


def train_core_dataset(core_loader, arch_id, criterion_id, optimizer_id, lr, num_epoch,
                       save_path=None, device_id=0, use_pretrained=True):
    device = 'cuda:{}'.format(device_id) if torch.cuda.is_available() else 'cpu'
    
    # init model
    model = None
    if arch_id == 'resnet18':
        if use_pretrained:
            model = resnet18(weights=ResNet18_Weights)
        else:
            model = resnet18()
    model = model.to(device)

    # init criterion
    criterion = None
    if criterion_id == 'ce':
        criterion = nn.CrossEntropyLoss()
    
    # init optimizer
    optimizer = None
    if optimizer_id == 'adam':
        optimizer = Adam(model.parameters(), lr=lr)
    
    # train
    running_loss = []
    running_acc = []
    for epoch in num_epoch:
        model.train()
        for iter_idx, (core_data, core_label) in enumerate(core_loader):
            core_data, core_label = core_data.to(device), core_label.to(device)

            optimizer.zero_grad()
            preds = model(core_data)
            loss = criterion(preds, core_label)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())

            if iter_idx == 0 or (iter_idx + 1) % 100 == 0 or (iter_idx + 1) == len(core_loader):
                print('#########epoch: {}, iter: {}, loss: {}#########'.format(epoch + 1, iter_idx + 1, loss.item())) 

        model.eval()
        with torch.no_grad():
            true_count = 0
            sample_count = 0
            for core_data, core_label in enumerate(core_loader):
                core_data, core_label = core_data.to(device), core_label.to(device)
                preds = model(core_data)
                
                predicted_label = torch.argmax(preds, dim=1)
                true_count += torch.count_nonzero(predicted_label == core_label).item()
                sample_count += core_data.shape[0]

            print('=========epoch: {}, accuracy: {}========='.format(epoch + 1, (true_count / sample_count)))
            running_acc.append((true_count / sample_count))
    
        if save_path:
            torch.save({
                'epoch': epoch, 
                'running_loss': running_loss,
                'running_acc': running_acc,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, save_path)

def train_train_dataset():
    pass