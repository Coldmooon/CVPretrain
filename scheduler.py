import math
import torch
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ChainedScheduler
from torch.optim.lr_scheduler import LambdaLR


# Define a custom class that inherits from LinearLR
class WarmupLR(LinearLR):
    # Override the constructor to accept an additional argument: end_factor
    def __init__(self, optimizer, start_factor, end_factor=1.0, total_iters=5, last_epoch=-1, verbose=False):
        # Pass the end_factor argument to the parent class constructor
        super().__init__(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=total_iters, last_epoch=last_epoch, verbose=verbose)

    # Override the _get_closed_form_lr method to use the end_factor
    def _get_closed_form_lr(self):
        return [base_lr * (self.start_factor + (self.end_factor - self.start_factor) * min(self.total_iters, self.last_epoch) / self.total_iters)
                for base_lr in self.base_lrs]


class Scheduler:
    def __init__(self, lr_policy, args):
        self.lr_policy = lr_policy
        self.args = args

    def create(self, optimizer, start_factor, end_factor=1, total_iters=5):
        if self.lr_policy == "StepWarmup":
            return self.StepWarmup(optimizer)
        if self.lr_policy == "CosWarmup":
            return self.CosinWarmup(optimizer, start_factor, end_factor, total_iters)
        else:
            raise ValueError("Unknow Learning Rate Policy...")


    def StepWarmup(self, optimizer):
        # Create an instance of the warmup scheduler with the desired parameters
        warmup_scheduler = WarmupLR(optimizer, start_factor=0.1/self.args.lr, total_iters=5, verbose=True)
        # Create an instance of the step decay scheduler with the desired parameters
        step_scheduler = MultiStepLR(optimizer, milestones=[60, 120, 150, 190], gamma=0.1, verbose=True)
        # Create an instance of the chained scheduler that combines the two schedulers
        scheduler = ChainedScheduler([warmup_scheduler, step_scheduler])

        return scheduler


    def CosinWarmup(self, optimizer, start_factor, end_factor, total_iters):
        lambda_cos = lambda epoch: (start_factor + (end_factor - start_factor) * epoch / total_iters)  if epoch <= total_iters else 0.5 * (1 + math.cos(math.pi * (epoch - total_iters) / (self.args.epochs - total_iters)))
        scheduler = LambdaLR(optimizer, lr_lambda=lambda_cos)

        return scheduler


    def learning_rate_planner(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10"""
        decay = 0
        if (self.args.dataset == 'imagenet'):
            decay = epoch // 30
        elif (self.args.dataset == 'cifar10' or self.args.dataset == 'cifar100'):
            decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
        elif (self.args.dataset == 'svhn'):
            decay = epoch % 48 == 0 and 1 or 0
        elif (self.args.dataset == 'mnist-rot-12k'):
            decay = epoch >= 225 and 2 or epoch >= 150 and 1 or 0
        else:
            print("No dataset named ", self.args.dataset)
            exit(-1)

        for param_group in optimizer.param_groups:
            # param_group['lr'] = lr # global learning rate
            param_group['lr'] = param_group['lr'] * torch.pow(0.1, decay) # learning rate for specific layer
