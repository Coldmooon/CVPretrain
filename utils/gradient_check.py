import torch

def gradient_check(optimizer, step_index):  
    # Checking gradients:
    with torch.no_grad():
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    if param.grad.dtype is torch.float16:
                        param.grad = param.grad.coalesce()
                    to_unscale = param.grad._values()
                else:
                    to_unscale = param.grad
                v = to_unscale.clone().abs().max()
                if torch.isinf(v) or torch.isnan(v):
                    print('INF in', group['layer_name'], 'of step', step_index, '!!!')

