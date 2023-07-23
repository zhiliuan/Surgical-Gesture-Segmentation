
import torch.nn as nn
from models.backbones.resnet import resnet18
from models.backbones.vgg import vgg16, vgg16_bn
from models.backbones import c3d
from models.backbones.r3d import r3d_18, r2plus1d_18
from models.attention import EncoderSelfAttention

backbone_dict = {'resnet': resnet18,
                 'vgg': vgg16, 'vgg_bn': vgg16_bn,
                 'c3d': c3d,
                 'r3d': r3d_18, 'r2plus1d': r2plus1d_18}

class _GestureTransformer(nn.Module):
    """Multi Modal model for gesture recognition on 3 channel"""
    def __init__(self, backbone: nn.Module, in_planes: int, out_planes: int,
                 pretrained: bool = False, dropout_backbone=0.1,
                 **kwargs):
        super(_GestureTransformer, self).__init__()

        self.in_planes = in_planes
        self.backbone = backbone(pretrained, in_planes, dropout=dropout_backbone)

        self.self_attention = EncoderSelfAttention(512, 64, 64, **kwargs)

        self.pool = nn.AdaptiveAvgPool2d((1, 512))
        self.classifier = nn.Linear(512, out_planes)
        # self.classifier = nn.Softmax(1)

    def forward(self, x,f):
        shape = x.shape
        x = x.view(-1, self.in_planes, x.shape[-2], x.shape[-1])
        x = self.backbone(x)
        x = x.view(shape[0], shape[1] // self.in_planes, -1)
        x = self.self_attention(x)
        x = self.pool(x).squeeze(dim=1)
        x = self.classifier(x)

        shape = f.shape
        f = f.view(-1, self.in_planes, f.shape[-2], f.shape[-1])
        f = self.backbone(f)
        f = f.view(shape[0], shape[1] // self.in_planes, -1)
        f = self.self_attention(f)
        f = self.pool(f).squeeze(dim=1)

        return x, f

def GestureTransoformer(backbone: str="resnet", in_planes: int=3, n_classes: int=15, **kwargs):
    if backbone not in backbone_dict:
        raise NotImplementedError("Backbone type: [{}] is not implemented.".format(backbone))
    model = _GestureTransformer(backbone_dict[backbone], in_planes, n_classes, **kwargs)
    return model