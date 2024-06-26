import argparse
import models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
        self.add_arguments()

    def add_arguments(self):
        # Data settings
        self.parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
                            help='path to dataset (default: imagenet)')
        self.parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
        self.parser.add_argument('--dataloader', default='pytorch', type=str,
                            help='Choose NVIDIA dali dataloader and use native PyTorch dataloader instead.')
        self.parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')
        
        # Training infrastructure
        self.parser.add_argument('--world-size', default=-1, type=int,
                            help='number of nodes for distributed training')
        self.parser.add_argument('--rank', default=-1, type=int,
                            help='node rank for distributed training')
        self.parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                            help='url used to set up distributed training')
        self.parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
        self.parser.add_argument('--seed', default=None, type=int,
                            help='seed for initializing training. ')
        self.parser.add_argument('--gpu', default=None, type=int,
                            help='GPU id to use.')
        self.parser.add_argument('--multiprocessing-distributed', action='store_true',
                            help='Use multi-processing distributed training to launch '
                                'N processes per node, which has N GPUs. This is the '
                                'fastest way to use PyTorch for either single node or '
                                'multi node data parallel training')
        self.parser.add_argument("--amp", action="store_true", 
                            help="Use torch.cuda.amp for mixed precision training")
        self.parser.add_argument('--compiled', default=0, type=int,
                            help='If use torch.compile, default is 0.')

        # Hyper-parameters
        self.parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                            choices=model_names,
                            help='model architecture: ' +
                                ' | '.join(model_names) +
                                ' (default: resnet18)')
        self.parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                            help='evaluate model on validation set')
        self.parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                            help='use pre-trained model')
        self.parser.add_argument('-b', '--batch-size', default=256, type=int,
                            metavar='N',
                            help='mini-batch size (default: 256), this is the total '
                                'batch size of all GPUs on the current node when '
                                'using Data Parallel or Distributed Data Parallel')
        self.parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        self.parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)',
                            dest='weight_decay')
        self.parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        self.parser.add_argument('--epochs', default=90, type=int, metavar='N',
                            help='number of total epochs to run')
        self.parser.add_argument('--optim', default='sgd', type=str, 
                            help='optimizer (default: sgd)')

        # Training tricks
        self.parser.add_argument('--ls', '--label-smoothing', default=0.0, type=float, dest='label_smoothing',
                            help='label smoothing')
        self.parser.add_argument('--gradclip', '--gradient-clip', default=0.0, type=float, dest='gradient_clip',
                            help='gradient clip')
        self.parser.add_argument('--startup-lr', default=0.0, type=float, dest='startup_lr',
                            help='set initial learning rate for warmup')
        self.parser.add_argument('--warmup-epochs', default=30, type=int, dest='warmup_epochs',
                            help='set how long is warmup')
        
        # Logging system
        self.parser.add_argument('--notes', metavar='DESCRIPTION', type=lambda s: {k:v for k,v in (i.split(':') for i in s.split(','))}, 
                            help="Comma-separated 'key:value' pairs, e.g. 'project_name:MyProject,notes:Some notes'")
        self.parser.add_argument('--logsys', metavar='LOG_SYSTEM', type=str,
                            help="Select log system, e.g., wandb or None")        
        self.parser.add_argument('--log-id', metavar='LOG_ID', type=str, dest='logid', 
                            help="A Wandb Run ID, e.g., sozupknt ")
        self.parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', 
                            help='print frequency (default: 10)')
        self.parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        self.parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')

    def parse_args(self):
        return self.parser.parse_args()
