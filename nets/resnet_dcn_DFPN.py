from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import math
import logging

import torch
import torch.nn as nn
from mmcv.ops import ModulatedDeformConv2dPack as DCN
import torch.utils.model_zoo as model_zoo

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :] 

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class Shuffle(nn.Module):
    def __init__(self, groups):
        super(Shuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups

        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)

        return x

class PoseResNet(nn.Module):

    def __init__(self, block, layers, heads,mid_channel,d_groups):
        self.inplanes = 64
        self.heads = heads
        self.deconv_with_bias = False
        self.deformable_groups = d_groups

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.DT5 = self._make_deconv_layer(2048,mid_channel[0],d_groups=self.deformable_groups)
        self.DT4 = self._make_deconv_layer(mid_channel[0],mid_channel[1],d_groups=self.deformable_groups)
        self.DT3 = self._make_deconv_layer(mid_channel[1],mid_channel[2],d_groups=self.deformable_groups)

        self.projectD4 = nn.Sequential(
            DCN(1024, mid_channel[0], kernel_size=(3, 3), stride=1, padding=1, dilation=1, deform_groups=self.deformable_groups),
            nn.BatchNorm2d(mid_channel[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.projectD3 = nn.Sequential(
            DCN(512, mid_channel[1], kernel_size=(3, 3), stride=1, padding=1, dilation=1,deform_groups=self.deformable_groups),
            nn.BatchNorm2d(mid_channel[1], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        for head in self.heads:
            classes = self.heads[head]
            fc = nn.Sequential(
            nn.Conv2d(mid_channel[2], mid_channel[2],kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel[2], classes, kernel_size=1, stride=1, padding=0, bias=True))
            if 'hm' or 'theta' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)

            self.__setattr__(head, fc)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, in_c, out_c, d_groups=1):
        layers = []

        kernel, padding, output_padding = self._get_deconv_cfg(4)

        fc = DCN(in_c, out_c, kernel_size=(3,3), stride=1,padding=1, dilation=1, deform_groups=d_groups)
        up = nn.ConvTranspose2d(
                in_channels=out_c,
                out_channels=out_c,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=self.deconv_with_bias,
                groups=1)
        fill_up_weights(up)

        layers.append(fc)
        layers.append(nn.BatchNorm2d(out_c, momentum=BN_MOMENTUM))
        layers.append(nn.ReLU(inplace=True))
        layers.append(up)
        layers.append(nn.BatchNorm2d(out_c, momentum=BN_MOMENTUM))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        feature = self.DT5(c5) + self.projectD4(c4) #(N,256,H/16,W/16)
        feature = self.DT4(feature) + self.projectD3(c3) #(N,128,H/8,W/8)
        feature = self.DT3(feature) #(N,64,H/4,W/4)

        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(feature)
        return [ret]

    def init_weights(self, num_layers):
        if 1:
            url = model_urls['resnet{}'.format(num_layers)]
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)

            print('=> init deconv weights from normal distribution')
            for m in [*self.DT5.modules(), *self.DT4.modules(), *self.DT3.modules(), *self.projectD4.modules(), *self.projectD3.modules()]:
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2],[256,128,64],1),
               34: (BasicBlock, [3, 4, 6, 3],[256,128,64],1),
               50: (Bottleneck, [3, 4, 6, 3],[256,128,64],1),
               101: (Bottleneck, [3, 4, 23, 3],[256,192,128],16),
               152: (Bottleneck, [3, 8, 36, 3],[256,256,256],16)}


def get_pose_net(num_layers, heads):
  block_class, layers, mid_channel,d_groups = resnet_spec[num_layers]

  model = PoseResNet(block_class, layers, heads,mid_channel,d_groups)
  model.init_weights(num_layers)
  return model

if __name__ == "__main__" :

    heads = {'hm': 80,
             'wh': 2 ,
             'reg': 2}

    model=get_pose_net(num_layers=50,heads=heads).cuda()
    print(model)
    x=torch.randn(1,3,512,512).cuda()
    out=model(x)
    print(out[0]['hm'].size())
    print(out[0]['wh'].size())

