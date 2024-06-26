import os
import random
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from opts import ArgumentParser
from datasets.dataloader import Dataloader
from optimizer import Optimizers
from train import Trainer
from checkpoints import Checkpoints
from model import Model
from scheduler import Scheduler
from utils import logger as Log

best_acc1 = 0
best_acc5 = 0

os.environ["WANDB__SERVICE_WAIT"] = "300"

def main():
    arg_parser = ArgumentParser()
    args = arg_parser.parse_args()

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        raise ValueError("If you want to enable training on MacOS, set device = torch.device('mps') in the main_worker().")
    else:
        raise ValueError("torch.cuda.device_count() returns False. Please check CUDA status.")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed


    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global best_acc5
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.gpu:
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device("cuda")

    # create modle
    model = Model.create(args, ngpus_per_node)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    
    # define optimizer
    optimizer = Optimizers.create(model, args.optim, args.lr, args.weight_decay, policy='no_bias_norm_decay')

    # define learning rate scheduler
    scheduler = Scheduler.create(optimizer, args.epochs, start_factor=args.startup_lr/args.lr, total_iters=args.warmup_epochs, lr_policy='CosWarmup')
    
    # define dataloader
    train_loader, val_loader = Dataloader.create(args, dataloader_type=args.dataloader)

    # construct Trainer
    Training = Trainer(model, optimizer, criterion, scheduler, args)

    if args.evaluate:
        Training.validate(val_loader)
        return

    # setup logger for rank 0
    logger = Log.Logger(args, model, ngpus_per_node)
    
    # optionally resume from a checkpoint
    checkpoints = Checkpoints(args, logger)
    best_acc1 = checkpoints.resume(model, optimizer, scheduler)

    is_best = None
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and args.dataloader=="pytorch":
            # train_sampler.set_epoch(epoch)
            train_loader.sampler.set_epoch(epoch)

        # train for one epoch
        Training.train(train_loader, epoch, logger)

        # evaluate on validation set
        acc1, acc5 = Training.validate(val_loader)
        
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)

        logger.log({"learning_rate": optimizer.param_groups[0]["lr"], "epoch": epoch, "top1.val": best_acc1, "top5.val": best_acc5})

        scheduler.step()

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            checkpoints.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best)
    
    logger.finish()


if __name__ == '__main__':
    main()
