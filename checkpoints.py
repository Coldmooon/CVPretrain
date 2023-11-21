import torch
import shutil
import os
import datetime

class Checkpoints:
    def __init__(self, args, logger):
        self.args = args
        self.savepath = self.setpath(logger)
        os.makedirs(self.savepath, exist_ok=True)
        os.makedirs(os.path.join(self.savepath, 'datasets'), exist_ok=True)
        os.makedirs(os.path.join(self.savepath, 'models'), exist_ok=True)
        self.experiment_files(['main.py', 
                               'model.py', 
                               'train.py', 
                               'opts.py', 
                               'optimizer.py', 
                               'scheduler.py', 
                               'models/resnetd.py',
                               'datasets/dataloader.py',
                               'datasets/transforms.py'
                               ])


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


    def setpath(self, logger=None):

        current_date = datetime.datetime.now().strftime("%H%M_%m%d%Y")
        hyper_parameters = f"{self.args.arch}_lr{self.args.lr}_epoch{self.args.epochs}_bs{self.args.batch_size * torch.cuda.device_count()}"
        dataset = logger.run.config['dataset'].replace(' ', '_') if logger and logger.run and logger.run.config.get('dataset') else ""
        project_path = logger.run.project.replace(' ', '_') if logger and logger.run and logger.run.project else "project.new.pytorch"
        description = logger.run.notes.replace(' ', '_') if logger and logger.run and logger.run.notes else ""

        components = [hyper_parameters, dataset, description, current_date]
        components = [c for c in components if c]

        experiment_path = '_'.join(components)

        # Use os.path.join to concatenate paths
        savepath = os.path.join('checkpoints', project_path, experiment_path)
        self.savepath = savepath

        return savepath


    def experiment_files(self, filelist):

        for file in filelist:
            shutil.copyfile(file, os.path.join(self.savepath, file))


    
    def save(self, state, is_best, filename='checkpoint.pth.tar'):
        
        filename = os.path.join(self.savepath, filename)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(self.savepath, 'model_best.pth.tar'))
