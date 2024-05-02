import torch
import logging

def test_mixed_linear(mixed_linear, test_loader, feature_backbone, params,
                      optimizer, running_test_acc, epoch, device, checkpoint):
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
    return running_test_acc, checkpoint

def train_accuracy_mixed_linear(mixed_linear, train_loader, feature_backbone, params,
                                running_train_acc, epoch, device, checkpoint):
    with torch.no_grad(): 
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
    return running_train_acc, checkpoint

def train_mixed_linear(mixed_linear, train_loader, feature_backbone, params,
                       optimizer, criterion, scheduler, running_loss, device,
                       epoch, checkpoint):
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
    return mixed_linear, optimizer, scheduler, running_loss, checkpoint