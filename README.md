# CVPretrain

CVPretrain is a project that optimizes the official PyTorch example on ImageNet. 

## Requirements

- PyTorch 2.0 or higher
- torchvision 0.16 or higher
- NVIDIA DALI 1.31 or higher

## Training

To run the main script, use the following command:

```bash
python main.py -a resnet50 [imagenet-folder with train and val folders]
```

### Single node, multiple GPUs

The following script will train a ResNet-50 model on ImageNet using the specified pre-trained model, warmup learning rate, mixed precision FP16, label-smoothing, mixup augmentation, and pytorch compiled model. For a full list of arguments, use `python main.py -h`.

```Single node, multiple GPUs:
python main.py -a resnet50 -b 512 --lr 0.2 -j 16 --label-smoothing 0.1 --compiled 1 --dist-url tcp://127.0.0.1:12000 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 [imagenet-folder with train and val folders]
```

### Multiple nodes:

Let Node 0 is the master of the distributed data parallel training. 

Node 0 IP address: 192.168.10.1
Port: 12000
batch-size: 1024

Node 0:

```bash
python main.py -a resnet50 -b 512 --lr 0.2 -j 16 --label-smoothing 0.1 --compiled 1 --dist-url tcp://127.0.0.1:12000 --dist-backend nccl --multiprocessing-distributed --world-size 2 --rank 0 [imagenet-folder with train and val folders]
```

Node 1:

```bash
python main.py -a resnet50 -b 512 --lr 0.2 -j 16 --label-smoothing 0.1 --compiled 1 --dist-url tcp://192.168.10.1:12000 --dist-backend nccl --multiprocessing-distributed --world-size 2 --rank 1 [imagenet-folder with train and val folders]
```

## Usage

```bash
usage: main.py [-h] [-a ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N] [--lr LR] [--momentum M] [--wd W] [-p N]
               [--resume PATH] [-e] [--pretrained] [--world-size WORLD_SIZE] [--rank RANK] [--dist-url DIST_URL]
               [--dist-backend DIST_BACKEND] [--seed SEED] [--gpu GPU] [--multiprocessing-distributed] [--dummy]
               [--compiled COMPILED] [--disable_dali] [--ls LABEL_SMOOTHING]
               [DIR]

PyTorch ImageNet Training

positional arguments:
  DIR                   path to dataset (default: imagenet)

options:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture: alexnet | convnext_base | convnext_large | convnext_small | convnext_tiny |
                        densenet121 | densenet161 | densenet169 | densenet201 | efficientnet_b0 | efficientnet_b1 |
                        efficientnet_b2 | efficientnet_b3 | efficientnet_b4 | efficientnet_b5 | efficientnet_b6 |
                        efficientnet_b7 | efficientnet_v2_l | efficientnet_v2_m | efficientnet_v2_s | get_model |
                        get_model_builder | get_model_weights | get_weight | googlenet | inception_v3 | list_models |
                        maxvit_t | mnasnet0_5 | mnasnet0_75 | mnasnet1_0 | mnasnet1_3 | mobilenet_v2 | mobilenet_v3_large |
                        mobilenet_v3_small | regnet_x_16gf | regnet_x_1_6gf | regnet_x_32gf | regnet_x_3_2gf | regnet_x_400mf
                        | regnet_x_800mf | regnet_x_8gf | regnet_y_128gf | regnet_y_16gf | regnet_y_1_6gf | regnet_y_32gf |
                        regnet_y_3_2gf | regnet_y_400mf | regnet_y_800mf | regnet_y_8gf | resnet101 | resnet152 | resnet18 |
                        resnet34 | resnet50 | resnext101_32x8d | resnext101_64x4d | resnext50_32x4d | shufflenet_v2_x0_5 |
                        shufflenet_v2_x1_0 | shufflenet_v2_x1_5 | shufflenet_v2_x2_0 | squeezenet1_0 | squeezenet1_1 | swin_b
                        | swin_s | swin_t | swin_v2_b | swin_v2_s | swin_v2_t | vgg11 | vgg11_bn | vgg13 | vgg13_bn | vgg16 |
                        vgg16_bn | vgg19 | vgg19_bn | vit_b_16 | vit_b_32 | vit_h_14 | vit_l_16 | vit_l_32 | wide_resnet101_2
                        | wide_resnet50_2 (default: resnet18)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when
                        using Data Parallel or Distributed Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --wd W, --weight-decay W
                        weight decay (default: 1e-4)
  -p N, --print-freq N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --world-size WORLD_SIZE
                        number of nodes for distributed training
  --rank RANK           node rank for distributed training
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend
  --seed SEED           seed for initializing training.
  --gpu GPU             GPU id to use.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is
                        the fastest way to use PyTorch for either single node or multi node data parallel training
  --dummy               use fake data to benchmark
  --compiled COMPILED   If use torch.compile, default is 0.
  --disable_dali        Disable DALI data loader and use native PyTorch one instead.
  --ls LABEL_SMOOTHING, --label-smoothing LABEL_SMOOTHING
                        label smoothing

```
