import torch
import shutil
import os

class Checkpoints:
    def __init__(self, args):
        self.args = args


    def resume(self, model, optimizer, scheduler):
        # optionally resume from a checkpoint
        best_acc1 = 0
        if self.args.resume:
            if os.path.isfile(self.args.resume):
                print("=> loading checkpoint '{}'".format(self.args.resume))
                if self.args.gpu is None:
                    checkpoint = torch.load(self.args.resume)
                elif torch.cuda.is_available():
                    # Map model to be loaded to specified single gpu.
                    loc = 'cuda:{}'.format(self.args.gpu)
                    checkpoint = torch.load(self.args.resume, map_location=loc)
                self.args.start_epoch = checkpoint['epoch']
                best_acc1 = checkpoint['best_acc1']
                if self.args.gpu is not None:
                    # best_acc1 may be from a checkpoint from a different GPU
                    best_acc1 = best_acc1.to(self.args.gpu)
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(self.args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.args.resume))
        else:
            print("=> args.resume: False. Train from scratch ...")

        return best_acc1


    def save(self, state, is_best=None, filename='checkpoint.pth.tar'):
        savepath = 'checkpoints'
        filename = savepath + '/' + filename
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, savepath + '/' + 'model_best.pth.tar')