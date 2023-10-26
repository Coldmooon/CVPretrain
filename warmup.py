import types
import math
from torch import inf
from functools import wraps, partial
import warnings
import weakref
from collections import Counter
from bisect import bisect_right

from .optimizer import Optimizer

from torch.optim.lr_scheduler import LRScheduler


# Define a custom class that inherits from LinearLR
class WarmupLR(LinearLR):
    # Override the constructor to accept an additional argument: end_factor
    def __init__(self, optimizer, start_factor, end_factor, total_iters, last_epoch=-1, verbose=False):
        self.end_factor = end_factor
        super().__init__(optimizer, start_factor, total_iters, last_epoch, verbose)

    # Override the _get_closed_form_lr method to use the end_factor
    def _get_closed_form_lr(self):
        return [base_lr * (self.start_factor + (self.end_factor - self.start_factor) * min(self.total_iters, self.last_epoch) / self.total_iters)
                for base_lr in self.base_lrs]

# Create an instance of the warmup scheduler with the desired parameters
warmup_scheduler = WarmupLR(optimizer, start_factor=0.01/0.05, end_factor=1.0, total_iters=5)

# Create an instance of the step decay scheduler with the desired parameters
step_scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

# Create an instance of the chained scheduler that combines the two schedulers
scheduler = ChainedScheduler([warmup_scheduler, step_scheduler])



class WarmupLR(LRScheduler):
    """Sets the learning rate of each parameter group using a linear warmup schedule, where :math:`\eta_{max}` is set to the base lr and :math:`T_{cur}` is the number of iterations since the start of warmup.

    Args:
        optimizer (Optimizer): Wrapped optimizer[^1^][1].
        warmup_iters (int): Number of iterations for the warmup phase.
        last_epoch (int): The index of last epoch[^2^][2][^3^][3]. Default: -1[^4^][4].
        verbose (bool): If ``True``, prints a message to stdout for each update[^5^][5][^6^][6]. Default: ``False``[^4^][4].

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.01 if epoch == 0 and iteration == 0
        >>> # lr = 0.02 if epoch == 0 and iteration == 1
        >>> # lr = 0.03 if epoch == 0 and iteration == 2
        >>> # lr = 0.04 if epoch == 0 and iteration == 3
        >>> # lr = 0.05 if epoch == 0 and iteration >= 4
        >>> # lr = 0.05 if epoch > 0 and epoch < 30
        >>> # lr = 0.005 if epoch >= 30 and epoch < 60
        >>> # lr = 0.0005 if epoch >= 60 and epoch < 90
        >>> scheduler = WarmupLR(optimizer, warmup_iters=4)
        >>> for epoch in range(100):
        >>>     for iteration in range(10):
        >>>         train(...)
        >>>         scheduler.step()
    """

    def __init__(self, optimizer, start_factor, end_factor, total_iters, last_epoch=-1, verbose=False):
        self.total_iters = total_iters
        self.start_factor = start_factor
        self.end_factor = end_factor
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning
            )
        
        # Compute the current iteration number
        iter_num = self.last_epoch + self.optimizer._step_count

        # Compute the linear increment factor for each iteration
        increment_factor = (1 - iter_num / self.warmup_iters) / (1 - (iter_num - 1) / self.warmup_iters)

        # Compute the learning rate using a linear function for warmup phase
        if iter_num < self.warmup_iters:
            return [base_lr + (group['lr'] - base_lr) * increment_factor for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)]
        
        # Compute the learning rate using a step function for normal phase
        else:
            return [group['lr'] * (0.1 ** ((self.last_epoch - self.warmup_iters) // 30)) for group in self.optimizer.param_groups]

