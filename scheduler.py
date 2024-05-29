import math
import torch
from torch.optim.lr_scheduler import LambdaLR,StepLR,CosineAnnealingLR,ExponentialLR

class Scheduler:
    def __init__(self, args, lr_policy='CosWarmup'):
        self.lr_policy = lr_policy
        self.args = args

    @classmethod
    def create(cls, optimizer, epochs, start_factor, end_factor=1, total_iters=5, milestones=[30, 60, 90], gamma=0.1, eta_min=0, lr_policy='CosWarmup'):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise ValueError("optimizer must be an instance of torch.optim.Optimizer")
        if total_iters < 0:
            raise ValueError("total_iters must be a non-negative integer")
        # ---------------------------------------------------------------------

        schedulers = {
            "steplr": lambda: StepLR(optimizer, step_size=milestones[0], gamma=gamma),
            "cosineannealinglr": lambda: CosineAnnealingLR(
                optimizer, T_max=epochs - total_iters, eta_min=eta_min
            ),
            "exponentiallr": lambda: ExponentialLR(optimizer, gamma=gamma),
            "StepWarmup": lambda: cls.StepWarmup(optimizer, start_factor, end_factor, total_iters, milestones, gamma),
            "CosWarmup": lambda: cls.CosinWarmup(optimizer, epochs, start_factor, end_factor, total_iters),
        }

        try:
            scheduler = schedulers[lr_policy]
            return scheduler()
        except KeyError:
            raise RuntimeError(
                f"Invalid lr scheduler '{lr_policy}'. Only StepWarmup, CosWarmup, StepLR, CosineAnnealingLR and ExponentialLR "
                "are supported."
            )


    @classmethod
    def StepWarmup(cls, optimizer, start_factor, end_factor, total_iters, milestones, gamma):
        # Define a lambda function that returns the learning rate factor
        # based on the current epoch and the warmup and decay parameters
        lambda_multi = lambda epoch: (start_factor + (end_factor - start_factor) * epoch / total_iters) if epoch <= total_iters else gamma ** len([m for m in milestones if m <= epoch])
        # Create a LambdaLR scheduler with the lambda function
        scheduler = LambdaLR(optimizer, lr_lambda=lambda_multi)

        return scheduler


    @classmethod
    def CosinWarmup(cls, optimizer, epochs, start_factor, end_factor, total_iters):
        lambda_cos = lambda epoch: (start_factor + (end_factor - start_factor) * epoch / total_iters)  if epoch <= total_iters else 0.5 * (1 + math.cos(math.pi * (epoch - total_iters) / (epochs - total_iters)))
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
