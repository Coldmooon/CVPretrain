import wandb

class Wanlog:
    def __init__(self, args, model, project, notes, tags, config):
        self.args = args
        self.watch_gradients = False
        self.run = self.setup(project, notes, tags, config)
        self.watch(model)  # watch gradients only for rank 0


    def watch(self, model):
        if self.run is not None and self.watch_gradients:
            self.run.watch(model)


    def log(self, info):
        if self.run is not None:
            self.run.log(info)

    
    def finish(self):
        if self.run is not None:
            self.run.finish()


    def setup(self, project, notes, tags, config):
        run = None
        if self.args.rank == 0:
            run = wandb.init(
                project = project,
                notes = notes,
                tags = tags,
                resume = self.args.resume and True or None,
                id = self.args.resume and self.args.logid or wandb.util.generate_id(),
                config = config
                )

        return run


class Logger:
    def __init__(self, args, model, ngpus_per_node):
        self.args = args
        self.watch_gradients = False
        self.project = self.args.notes.get('project_name') if self.args.notes and 'project_name' in self.args.notes else "pytorch.examples.ddp"
        self.notes = self.args.notes.get('notes') if self.args.notes and 'notes' in self.args.notes else ''
        self.tags = None
        self.config = {
                        "architecture": self.args.arch,
                        "learning_rate": self.args.lr,
                        "label_smoothing": self.args.label_smoothing,
                        "epochs": self.args.epochs,
                        "batch_size": self.args.batch_size * ngpus_per_node,
                        "dataset": "imagenet",
                        "num_workers": self.args.workers * ngpus_per_node,
                        "num_gpus": self.args.world_size,
                        "world-size": self.args.world_size / ngpus_per_node,
                        "compiled": self.args.compiled,
                        "mixed precision": 1, 
                      }
        self.logger = self.setlogger(model, args.logsys)


    def setlogger(self, model, logsystem):
        if (logsystem is None or self.args.rank != 0):
            print("No log system used...")
        elif (logsystem == 'wandb'):
            print("Using wandb log.")
            return Wanlog(self.args, model, self.project, self.notes, self.tags, self.config)

        return None
    

    def watch(self, model):
        if self.logger is not None and self.watch_gradients:
            self.logger.watch(model)


    def log(self, info):
        if self.logger is not None:
            self.logger.log(info)

    
    def finish(self):
        if self.logger is not None:
            self.run.finish()
