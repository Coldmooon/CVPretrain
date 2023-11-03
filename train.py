import time
import math
from enum import Enum
import torch
import torch.distributed as dist
import torch.cuda.amp as amp
from torch.utils.data import Subset
from datasets import transforms as Localtrans


def train(train_loader, model, criterion, optimizer, epoch, device, args, run=None, scaler=None, do_log=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.disable_dali:
            images, target = data
        else:
            images = data[0]["data"]
            target = data[0]["label"].squeeze(-1).long()

        images, target = Localtrans.datamix(images, target)

        # # move data to the same device as model
        # images = images.to(device, non_blocking=True)
        # target = target.to(device, non_blocking=True)

        # compute output
        # Wrap the model and optimizer in autocast context manager
        with amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        
        # Use GradScaler to unscale and update the gradients
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)
        
        if do_log:
            run.log({"data_time": data_time.val, "batch_loss": losses.avg, "batch_time": batch_time.val, "top1.train":top1.avg, "top5.train":top5.avg})


def validate(val_loader, model, criterion, args):

    if args.disable_dali:
        if args.distributed:
            val_loader_len = len(val_loader.sampler) 
        else:
            val_loader_len = len(val_loader)
    else:
        val_loader_len = int(math.ceil(val_loader._size / args.batch_size))

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()

            for i, data in enumerate(loader):
                if args.disable_dali:
                    images, target = data
                else:
                    images = data[0]["data"]
                    target = data[0]["label"].squeeze(-1).long()
                    
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    # images = images.cuda(args.gpu, non_blocking=True)
                    pass # just pass for DALI
                if torch.backends.mps.is_available():
                    images = images.to('mps')
                    target = target.to('mps')
                if torch.cuda.is_available():
                    # target = target.cuda(args.gpu, non_blocking=True)
                    pass # just pass for DALI

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        # len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        val_loader_len,
        [batch_time, losses, top1, top5],
        prefix='Test: ')    

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if not args.disable_dali:
        # # TO DO: for DALI: ensure that in distributed training mode, each process can evaluate the entire validation set, instead of only a part of the data.
        pass 
    else:
        # Distributed PyTorch validation
        if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
            aux_val_dataset = Subset(val_loader.dataset,
                                    range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
            aux_val_loader = torch.utils.data.DataLoader(
                aux_val_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
            run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()
    
    return top1.avg, top5.avg

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Check the dimension of the target
        if target.dim() == 1:
            # Target is already 1-D, use it directly
            pass
        elif target.dim() == 2:
            # Target is 2-D, get the indices by using torch.max
            target = torch.argmax(target, dim=1)
        else:
            # Target has an invalid dimension, raise an error
            raise ValueError("Target should be either 1-D or 2-D")

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res