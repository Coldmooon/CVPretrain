from torchvision.transforms import v2

def datamix(images, target):
    # cutmix = v2.CutMix(num_classes=1000)
    mixup = v2.MixUp(alpha=0.2, num_classes=1000)
    # cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    # converted to one-hot targets
    images, target = mixup(images, target)
    return images, target
