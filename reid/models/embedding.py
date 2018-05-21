import math
import copy
from torch import nn
import torch
import torch.nn.functional as F
import pdb


class RandomWalkEmbed(nn.Module):
    def __init__(self, instances_num=4, feat_num=2048, num_classes=0):
        super(RandomWalkEmbed, self).__init__()
        self.instances_num = instances_num
        self.feat_num = feat_num
        self.bn = nn.BatchNorm1d(feat_num)
        self.classifier = nn.Linear(feat_num, num_classes)
        self.classifier.weight.data.normal_(0, 0.001)
        self.classifier.bias.data.zero_()
        

    def forward(self, probe_x, gallery_x):
        probe_x.contiguous()
        gallery_x.contiguous()
        N_probe = probe_x.size(0)
        N_gallery = gallery_x.size(0)

        probe_x = probe_x.unsqueeze(1)
        probe_x = probe_x.expand(N_probe, N_gallery, self.feat_num)
        probe_x = probe_x.contiguous()

        gallery_x = gallery_x.unsqueeze(0)
        gallery_x = gallery_x.expand(N_probe, N_gallery, self.feat_num)
        gallery_x = gallery_x.contiguous()

        diff = torch.pow(probe_x - gallery_x, 2)
        diff = diff.view(N_probe * N_gallery, -1)
        diff = diff.contiguous()
        bn_diff = self.bn(diff)

        cls_encode = self.classifier(bn_diff)
        cls_encode = cls_encode.view(N_probe, N_gallery, -1)

        return cls_encode


class EltwiseSubEmbed(nn.Module):
    def __init__(self, nonlinearity='square', use_batch_norm=False,
                 use_classifier=False, num_features=0, num_classes=0):
        super(EltwiseSubEmbed, self).__init__()
        self.nonlinearity = nonlinearity
        if nonlinearity is not None and nonlinearity not in ['square', 'abs']:
            raise KeyError("Unknown nonlinearity:", nonlinearity)
        self.use_batch_norm = use_batch_norm
        self.use_classifier = use_classifier
        if self.use_batch_norm:
            self.bn = nn.BatchNorm1d(num_features)
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()
        if self.use_classifier:
            assert num_features > 0 and num_classes > 0
            self.classifier = nn.Linear(num_features, num_classes)
            self.classifier.weight.data.normal_(0, 0.001)
            self.classifier.bias.data.zero_()

    def forward(self, x1, x2):
        x = x1 - x2
        if self.nonlinearity == 'square':
            x = x.pow(2)
        elif self.nonlinearity == 'abs':
            x = x.abs()
        if self.use_batch_norm:
            x = self.bn(x)
        if self.use_classifier:
            x = x.view(x.size(0),-1)
            x = self.classifier(x)
        else:
            x = x.sum(1)
        return x

