import torch
import shutil
import os
import datetime

class Checkpoints:
    def __init__(self, args):
        self.args = args


    def resume(self, model, optimizer, scheduler):
        # optionally resume from a checkpoint
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
                    # best_acc1 should be a torch.Tensor since it is loaded by torch.load().
                    # best_acc1 = best_acc1.to(self.args.gpu)
                    pass
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(self.args.resume, checkpoint['epoch']))
                return best_acc1
            else:
                print("=> no checkpoint found at '{}'".format(self.args.resume))
        else:
            print("=> args.resume: False. Train from scratch ...")

        return 0


    def save(self, state, is_best=None, logger=None, filename='checkpoint.pth.tar'):
        current_date = datetime.date.today().strftime("%m_%d_%Y") 
        description = self.args.arch + "_" + \
                      self.args.lr + "_" + \
                      self.args.epochs + "_" + \
                      self.args.batch_size * torch.cuda.device_count()       
        project = logger and '/' + logger.project or ""
        note = logger and logger.notes or ""
        dataset = logger and logger.config['dataset'] or ""
        savefolder = description + '_' + dataset + '_' + note + '_' + current_date
        
        savepath = 'checkpoints' + project + '/' + savefolder
        filename = savepath + '/' + filename
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, savepath + '/' + 'model_best.pth.tar')