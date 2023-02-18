'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.unet_parts import *

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out

class DenseNet(nn.Module):
    def __init__(self, input_bands, block, nblocks, growth_rate=12, reduction=0.5, num_classes=3):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(input_bands, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate
        self.trans4 = Transition(1024, 1024)

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        x1 = self.trans1(self.dense1(out))
        x2 = self.trans2(self.dense2(x1))
        x3 = self.trans3(self.dense3(x2))
        x4 = self.trans4(self.dense4(x3))
        return x4

class densenet_aspp(nn.Module):
    def __init__(self, input_bands, num_classes):
        super(densenet_aspp, self).__init__()
        self.feature_model = DenseNet(input_bands, Bottleneck, [6,12,24,16], growth_rate=32)
        self.conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)
        bilinear = True
        self.up1 = Up(512, 256, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(512, 256, bilinear)
        self.up4 = Up(512, 256, bilinear)
        self.aspp = aspp()

        self.out = nn.Conv2d(256, num_classes, 3, 1, 1)
        self.softmax = nn.Softmax()


    def forward(self, x):
        x4 = self.feature_model(x)
        p4 = self.conv4(x4)
        p4 = self.aspp(p4)
        out = self.out(p4)
        out = F.interpolate(out, scale_factor=16, mode='bilinear')
        out = self.softmax(out)
        return out

class Atrous_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(Atrous_module, self).__init__()
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=3,
                                            stride=1, padding=rate, dilation=rate)
        self.batch_norm = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.batch_norm(x)
        return x


class aspp(nn.Module):
    def __init__(self):
        super(aspp, self).__init__()
        rates = [1, 3, 6, 12]
        self.aspp1 = Atrous_module(256, 256, rate=rates[0])
        self.aspp2 = Atrous_module(256, 256, rate=rates[1])
        self.aspp3 = Atrous_module(256, 256, rate=rates[2])
        self.aspp4 = Atrous_module(256, 256, rate=rates[3])
        self.image_pool = nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                        nn.Conv2d(256, 256, kernel_size=1))

        self.fc1 = nn.Sequential(nn.Conv2d(1280, 256, kernel_size=1),
                                 nn.BatchNorm2d(256))
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.image_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='nearest')

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.relu(self.bn(self.fc1(x)))
        return x


# def DenseNet121():
#     return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)
#
# def DenseNet169():
#     return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)
#
# def DenseNet201():
#     return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)
#
# def DenseNet161():
#     return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)
#
# def densenet_cifar():
#     return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12)
#
# def test():
#     net = densenet_aspp(2)
#     x = torch.randn(1,3,64,64)
#     y = net(x)
#     print(y.shape)
#
# test()
