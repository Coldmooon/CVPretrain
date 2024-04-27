from .resnetd import * # restricted by __all__ in resnetd.py
from models.external.vit_pytorch.vit_pytorch.vit import ViT
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, resnext101_64x4d , shufflenet_v2_x0_5 , shufflenet_v2_x1_0 , shufflenet_v2_x1_5 , shufflenet_v2_x2_0 , squeezenet1_0 , squeezenet1_1 , swin_b , swin_s , swin_t , swin_v2_b , swin_v2_s , swin_v2_t , vgg11 , vgg11_bn , vgg13 , vgg13_bn , vgg16 , vgg16_bn , vgg19 , vgg19_bn , vit_b_16 , vit_b_32 , vit_h_14 , vit_l_16 , vit_l_32 , wide_resnet101_2 , wide_resnet50_2


def vit_pytorch_base_patch32():
    return ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )