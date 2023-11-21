import wandb

class Wanlog:
    def __init__(self, args, model, ngpus_per_node):
        self.args = args
        self.watch_gradients = False
        self.run = self.setup(ngpus_per_node)
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


    def setup(self, ngpus_per_node):
        run = None
        if self.args.rank == 0:
            run = wandb.init(
                project = self.args.notes.get('project_name') if self.args.notes and 'project_name' in self.args.notes else "pytorch.examples.ddp",
                notes = self.args.notes.get('notes') if self.args.notes and 'notes' in self.args.notes else '',
                tags = None,
                resume = self.args.resume and True or None,
                id = self.args.resume and self.args.logid or wandb.util.generate_id(),
                config={
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
                )

        return run


class Logger:
    def __init__(self, args, model, ngpus_per_node, logsystem='wandb'):
        self.args = args
        self.logger = self.setlogger(model, ngpus_per_node, logsystem)


    def setlogger(self, model, ngpus_per_node, logsystem):
        if (logsystem == 'wandb'):
            return Wanlog(self.args, model, ngpus_per_node)

        return None