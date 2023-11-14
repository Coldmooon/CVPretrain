import wandb

class wanlog:
    def __init__(self, args, ngpus_per_node):
        self.args = args
        self.run = self.setup(ngpus_per_node)

    def setup(self, ngpus_per_node):
        run = None
        if self.args.rank == 0:
            run = wandb.init(
                project="pytorch.examples.ddp",
                notes = self.args.notes,
                tags = None,
                config={
                        "architecture": self.args.arch,
                        "learning_rate": self.args.lr,
                        "label_smoothing": self.args.label_smoothing,
                        "epochs": self.args.epochs,
                        "batch_size": self.args.batch_size * self.args.world_size / ngpus_per_node,
                        "dataset": "ImageNet 2012",
                        "num_workers": self.args.workers,
                        "num_gpus": self.args.world_size,
                        "world-size": self.args.world_size / ngpus_per_node,
                        "compiled": self.args.compiled,
                        "mixed precision": 1, 
                        }
                )

        return run
    
    def watch(self, model):
        if self.run is not None:
            self.run.watch(model)


    def log(self, info):
        if self.run is not None:
            self.run.log(info)

    
    def finish(self):
        if self.run is not None:
            self.run.finish()