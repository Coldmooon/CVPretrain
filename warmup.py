from torch.optim.lr_scheduler import LinearLR

# Define a custom class that inherits from LinearLR
class WarmupLR1(LinearLR):
    # Override the constructor to accept an additional argument: end_factor
    def __init__(self, optimizer, start_factor, end_factor=1.0, total_iters=5, last_epoch=-1, verbose=False):
        # Pass the end_factor argument to the parent class constructor
        super().__init__(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=total_iters, last_epoch=last_epoch, verbose=verbose)

    # Override the _get_closed_form_lr method to use the end_factor
    def _get_closed_form_lr(self):
        return [base_lr * (self.start_factor + (self.end_factor - self.start_factor) * min(self.total_iters, self.last_epoch) / self.total_iters)
                for base_lr in self.base_lrs]
