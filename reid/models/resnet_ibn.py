from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch

from .resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a, resnet152_ibn_a
from .resnet_ibn_b import resnet50_ibn_b, resnet101_ibn_b, resnet152_ibn_b


__all__ = ['ResNetIBN', 'resnet_ibn50a', 'resnet_ibn101a', 'resne_ibn152a',
            'resnet_ibn50b', 'resnet_ibn101b', 'resne_ibn152b']


class ResNetIBN(nn.Module):
    __factory = {
        '50a': resnet50_ibn_a,
        '101a': resnet101_ibn_a,
        '152a': resnet152_ibn_a,
        '50b': resnet50_ibn_b,
        '101b': resnet101_ibn_b,
        '152b': resnet152_ibn_b,
    }

    def __init__(self, depth, pretrained=False, cut_at_pooling=False,
                 num_features=256, norm=False, dropout=0.5, num_classes=0):
        super(ResNetIBN, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.eps = 1e-12
        # Construct base (pretrained) resnet
        if depth not in ResNetIBN.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNetIBN.__factory[depth](pretrained=pretrained)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = self.base.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=True)
                init.normal_(self.classifier.weight, std=0.001)
                init.constant_(self.classifier.bias, 0)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        if self.cut_at_pooling:
            return x

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        if self.has_embedding:
            x = self.feat(x)
        x = x.view(x.size(0), x.size(1), 1)
        x = self.feat_bn(x)
        x = x.view(x.size(0), x.size(1))
        if self.training is False:
            x = F.normalize(x)
            return x
        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)
        return x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def resnet_ibn50a(**kwargs):
    return ResNetIBN('50a', **kwargs)


def resnet_ibn101a(**kwargs):
    return ResNetIBN('101a', **kwargs)


def resne_ibn152a(**kwargs):
    return ResNetIBN('152a', **kwargs)


def resnet_ibn50b(**kwargs):
    return ResNetIBN('50b', **kwargs)


def resnet_ibn101b(**kwargs):
    return ResNetIBN('101b', **kwargs)


def resne_ibn152b(**kwargs):
    return ResNet('152b', **kwargs)