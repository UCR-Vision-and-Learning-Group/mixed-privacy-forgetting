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



