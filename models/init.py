import torch
import torchvision.models as models

class Model:
    def __init__(self, args):
        self.args = args


    def create(self, ngpus_per_node):
        model = self.setup(ngpus_per_node)
        return model


    def setup(self, ngpus_per_node, model=None):

        if self.args.pretrained:
            print("=> using pre-trained model '{}'".format(self.args.arch))
            model = models.__dict__[self.args.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(self.args.arch))
            model = models.__dict__[self.args.arch]()

        if self.args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if self.args.gpu is not None:
                print('args.gpu:::::::', self.args.gpu)
                torch.cuda.set_device(self.args.gpu)
                # Do not use `torch.cuda.device(self.args.gpu)`, otherwise will lead to NCCL Errors.
                model.cuda(self.args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                self.args.batch_size = int(self.args.batch_size / ngpus_per_node)
                self.args.workers = int((self.args.workers + ngpus_per_node - 1) / ngpus_per_node)
                if self.args.compiled == 1:
                    model = torch.compile(model)   

                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.args.gpu])
                torch.cuda.current_stream().wait_stream(s)
            else:
                model.cuda()
                if self.args.compiled == 1:
                    model = torch.compile(model)   
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
        elif self.args.gpu is not None and torch.cuda.is_available():
            torch.cuda.set_device(self.args.gpu)
            # Do not use `torch.cuda.device(self.args.gpu)`, otherwise will lead to NCCL Errors.
            model = model.cuda(self.args.gpu)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            if self.args.arch.startswith('alexnet') or self.args.arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features)
                model.cuda()
            else:
                model = torch.nn.DataParallel(model).cuda()

        return model
