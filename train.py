import time
import math
from enum import Enum
import torch
import torch.distributed as dist
import torch.cuda.amp as amp
from torch.utils.data import Subset
from datasets import transforms as Localtrans

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


# Define any helper functions or classes that are used by the methods
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
        if self.summary_type == Summary.NONE:
            fmtstr = ''
        elif self.summary_type == Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type == Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type == Summary.COUNT:
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


class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler, args):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.args = args
        print("--------------------------------------------------------------------")
        print("Support BF16?", torch.cuda.is_bf16_supported())
        print("torch.backends.cudnn.allow_tf32: ", torch.backends.cudnn.allow_tf32)
        print("torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction", torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        print("--------------------------------------------------------------------")
        self.scaler = amp.GradScaler()
        self.gradient_clip = args.gradient_clip
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def train(self, train_loader, epoch, logger):

        self.model.train()

        # Initialize the meters to track the metrics
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))       

        # Start the timer
        end = time.time()

        # Loop over the batches
        for i, data in enumerate(train_loader):
            # Measure data loading time
            data_time.update(time.time() - end)

            if self.args.dataloader == 'pytorch':
                images, target = data
                # move data to the same device as model
                images = images.to(self.args.gpu, non_blocking=True)
                target = target.to(self.args.gpu, non_blocking=True)
            else:
                images = data[0]["data"]
                target = data[0]["label"].squeeze(-1).long()

            images, target = Localtrans.datamix(images, target)

            # compute output
            # Wrap the model and optimizer in autocast context manager
            with amp.autocast(enabled=self.args.amp, dtype=self.dtype):
                output = self.model(images)
                # print(f"Expected dtype: {self.dtype}, Actual dtype: {output.dtype}")  # Debug print
                # output is float16 because linear layers autocast to float16.
                assert output.dtype == self.dtype

                loss = self.criterion(output, target)
                # loss layers autocast to float32.
                assert loss.dtype == torch.float32

            # Measure the accuracy and record the loss
            acc1, acc5 = self.accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # Use GradScaler to unscale and update the gradients
            self.scaler.scale(loss).backward()
            
            # Gradient Clip
            if self.gradient_clip > 1e-3:
                # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
                # Unscales the gradients of optimizer's assigned params in-place
                self.scaler.unscale_(self.optimizer)
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)   

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.optimizer.zero_grad()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Print the progress every print_freq batches
            if i % self.args.print_freq == 0:
                progress.display(i)
            if i % self.args.print_freq == 0 and logger is not None:
                logger.log({"data_time": data_time.val, "batch_loss": losses.avg, "batch_time": batch_time.val, "top1.train":top1.avg, "top5.train":top5.avg})


    def validate(self, val_loader):

        self.model.eval()

        if self.args.dataloader == 'pytorch':
            if self.args.distributed:
                val_loader_len = len(val_loader.sampler) 
            else:
                val_loader_len = len(val_loader)
        else:
            val_loader_len = int(math.ceil(val_loader._size / self.args.batch_size))

        # Initialize the meters to track the metrics  
        batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        losses = AverageMeter('Loss', ':.4e', Summary.NONE)
        top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
        progress = ProgressMeter(
            val_loader_len,
            [batch_time, losses, top1, top5],
            prefix='Test: ')  
        
        self.run_validate(val_loader, batch_time, losses, top1, top5, progress)

        if self.args.distributed:
            top1.all_reduce()
            top5.all_reduce()

        if self.args.dataloader == 'dali':
            # # TO DO: for DALI: ensure that in distributed training mode, each process can evaluate the entire validation set, instead of only a part of the data.
            pass 
        elif self.args.dataloader == 'pytorch':
            # Distributed PyTorch validation
            if self.args.distributed and (len(val_loader.sampler) * self.args.world_size < len(val_loader.dataset)):
                aux_val_dataset = Subset(val_loader.dataset,
                                        range(len(val_loader.sampler) * self.args.world_size, len(val_loader.dataset)))
                aux_val_loader = torch.utils.data.DataLoader(
                    aux_val_dataset, batch_size=self.args.batch_size, shuffle=False,
                    num_workers=self.args.workers, pin_memory=True)
                self.run_validate(aux_val_loader, batch_time, losses, top1, top5, progress, len(val_loader))
        else:
            raise NotImplementedError

        progress.display_summary()
        
        return top1.avg, top5.avg
    

    def run_validate(self, val_loader, batch_time, losses, top1, top5, progress, base_progress=0):

        with torch.no_grad():

            end = time.time()

            for i, data in enumerate(val_loader):
                if self.args.dataloader == 'pytorch':
                    images, target = data
                else:
                    images = data[0]["data"]
                    target = data[0]["label"].squeeze(-1).long()
                    
                i = base_progress + i
                if self.args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(self.args.gpu, non_blocking=True)
                    # pass # just pass for DALI
                if torch.backends.mps.is_available():
                    images = images.to('mps')
                    target = target.to('mps')
                if torch.cuda.is_available():
                    target = target.cuda(self.args.gpu, non_blocking=True)
                    # pass # just pass for DALI


                # compute output
                output = self.model(images)
                loss = self.criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = self.accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.args.print_freq == 0:
                    progress.display(i + 1)


        # Return the validation loss and accuracy
        return top1, top5, losses, progress


    def accuracy(self, output, target, topk=(1,)):
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



