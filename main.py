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
from datasets import dataloader as Dataloader
from optimizer import Optimizers
from train import Trainer
from checkpoints import Checkpoints
from model import Model
from scheduler import Scheduler
from utils import logger

best_acc1 = 0
best_acc5 = 0

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

    # create modle
    modeling = Model(args)
    model = modeling.create(ngpus_per_node)

    if args.gpu:
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device("cuda")

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    
    # define optimizer
    optimzers = Optimizers(args)
    optimizer = optimzers.create(model, policy='regular')

    # define learning rate scheduler
    scheduling = Scheduler(args, lr_policy='CosWarmup')
    scheduler = scheduling.create(optimizer, start_factor=0.1/args.lr, total_iters=5)

    # define dataloader
    train_loader, val_loader = Dataloader.dataloader(args)

    # construct Trainer
    Training = Trainer(model, optimizer, criterion, scheduler, args)

    if args.evaluate:
        Training.validate(val_loader)
        return

    # setup logger for rank 0
    wanlog = logger.Logger(args, model, ngpus_per_node)
    wanlog = wanlog.logger
    
    # optionally resume from a checkpoint
    checkpoints = Checkpoints(args, wanlog)
    best_acc1 = checkpoints.resume(model, optimizer, scheduler)

    wanlog.log({"learning_rate": optimizer.param_groups[0]["lr"]})

    is_best = None
    checkpoints.save({
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and args.disable_dali:
            # train_sampler.set_epoch(epoch)
            train_loader.sampler.set_epoch(epoch)

        # train for one epoch
        Training.train(train_loader, epoch, wanlog)

        # evaluate on validation set
        acc1, acc5 = Training.validate(val_loader)
        
        scheduler.step()
        
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)

        wanlog.log({"learning_rate": optimizer.param_groups[0]["lr"], "epoch": epoch, "top1.val": best_acc1, "top5.val": best_acc5})

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
    
    wanlog.finish()


if __name__ == '__main__':
    main()
