import torch.nn as nn
import torch
import torch.nn as nn
import torchvision

class Conv2dBn(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(Conv2dBn, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        return self.conv(x)


class Conv2dRelu(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(Conv2dRelu, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=bias),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Conv2dBnRelu(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(Conv2dBnRelu, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


'''
	与torchvision.models.resnet中的BasicBlock不同，
	其没有dilation参数,无法组成Dilated ResNet
'''


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()

        self.conv1 = Conv2dBnRelu(in_ch, out_ch, kernel_size=3,
                                  stride=stride, padding=dilation, dilation=dilation, bias=False)

        self.conv2 = Conv2dBn(in_ch, out_ch, kernel_size=3,
                              stride=1, padding=dilation, dilation=dilation, bias=False)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()

        self.conv1 = Conv2dBnRelu(in_ch, out_ch, kernel_size=1, bias=False)

        self.conv2 = Conv2dBnRelu(out_ch, out_ch, kernel_size=3, stride=stride,
                                  padding=dilation, dilation=dilation, bias=False)

        self.conv3 = Conv2dBn(out_ch, out_ch * 4, kernel_size=1, bias=False)

        self.downsample = downsample

        self.relu = nn.ReLU(inplace=True)

        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

'''
Global Attention Upsample Module
'''


class GAUModule(nn.Module):
    def __init__(self, in_ch, out_ch):  #
        super(GAUModule, self).__init__()

        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2dBn(out_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.conv2 = Conv2dBnRelu(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

    # x: low level feature
    # y: high level feature
    def forward(self, x, y):
        h, w = x.size(2), x.size(3)
        y_up = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(y)
        x = self.conv2(x)
        y = self.conv1(y)
        z = torch.mul(x, y)

        return y_up + z


'''
Feature Pyramid Attention Module
FPAModule1:
	downsample use maxpooling
'''


class FPAModule1(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(FPAModule1, self).__init__()

        # global pooling branch
        self.branch1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )

        # midddle branch
        self.mid = nn.Sequential(
            Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )

        self.down1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dBnRelu(in_ch, 1, kernel_size=7, stride=1, padding=3)
        )

        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dBnRelu(1, 1, kernel_size=5, stride=1, padding=2)
        )

        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dBnRelu(1, 1, kernel_size=3, stride=1, padding=1),
            Conv2dBnRelu(1, 1, kernel_size=3, stride=1, padding=1),
        )

        self.conv2 = Conv2dBnRelu(1, 1, kernel_size=5, stride=1, padding=2)
        self.conv1 = Conv2dBnRelu(1, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        b1 = self.branch1(x)
        b1 = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(b1)

        mid = self.mid(x)

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3 = nn.Upsample(size=(h // 4, w // 4), mode='bilinear', align_corners=True)(x3)

        x2 = self.conv2(x2)
        x = x2 + x3
        x = nn.Upsample(size=(h // 2, w // 2), mode='bilinear', align_corners=True)(x)

        x1 = self.conv1(x1)
        x = x + x1
        x = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(x)

        x = torch.mul(x, mid)
        x = x + b1
        return x


'''
Feature Pyramid Attention Module
FPAModule2:
	downsample use convolution with stride = 2
'''


class FPAModule2(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(FPAModule2, self).__init__()

        # global pooling branch
        self.branch1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )

        # midddle branch
        self.mid = nn.Sequential(
            Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )

        self.down1 = Conv2dBnRelu(in_ch, 1, kernel_size=7, stride=2, padding=3)

        self.down2 = Conv2dBnRelu(1, 1, kernel_size=5, stride=2, padding=2)

        self.down3 = nn.Sequential(
            Conv2dBnRelu(1, 1, kernel_size=3, stride=2, padding=1),
            Conv2dBnRelu(1, 1, kernel_size=3, stride=1, padding=1),
        )

        self.conv2 = Conv2dBnRelu(1, 1, kernel_size=5, stride=1, padding=2)
        self.conv1 = Conv2dBnRelu(1, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        b1 = self.branch1(x)
        b1 = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(b1)

        mid = self.mid(x)

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3 = nn.Upsample(size=(h // 4, w // 4), mode='bilinear', align_corners=True)(x3)

        x2 = self.conv2(x2)
        x = x2 + x3
        x = nn.Upsample(size=(h // 2, w // 2), mode='bilinear', align_corners=True)(x)

        x1 = self.conv1(x1)
        x = x + x1
        x = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(x)

        x = torch.mul(x, mid)
        x = x + b1
        return x


'''
papers:
	Pyramid Attention Networks
'''


class PAN(nn.Module):
    def __init__(self, backbone, pretrained=True, n_class=2):
        '''
        :param backbone: Bcakbone network
        '''
        super(PAN, self).__init__()

        if backbone.lower() == 'resnet34':
            encoder = torchvision.models.resnet34(pretrained)
            bottom_ch = 512
        elif backbone.lower() == 'resnet50':
            encoder = torchvision.models.resnet50(pretrained)
            bottom_ch = 2048
        elif backbone.lower() == 'resnet101':
            encoder = torchvision.models.resnet101(pretrained)
            bottom_ch = 2048
        elif backbone.lower() == 'resnet152':
            encoder = torchvision.models.resnet152(pretrained)
            bottom_ch = 2048
        else:
            raise NotImplementedError('{} Backbone not implement'.format(backbone))

        self.conv1 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool)  # 1/4
        self.conv2_x = encoder.layer1  # 1/4
        self.conv3_x = encoder.layer2  # 1/8
        self.conv4_x = encoder.layer3  # 1/16
        self.conv5_x = encoder.layer4  # 1/32

        self.fpa = FPAModule1(in_ch=bottom_ch, out_ch=n_class)

        self.gau3 = GAUModule(in_ch=bottom_ch // 2, out_ch=n_class)

        self.gau2 = GAUModule(in_ch=bottom_ch // 4, out_ch=n_class)

        self.gau1 = GAUModule(in_ch=bottom_ch // 8, out_ch=n_class)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x1 = self.conv1(x)
        x2 = self.conv2_x(x1)
        x3 = self.conv3_x(x2)
        x4 = self.conv4_x(x3)
        x5 = self.conv5_x(x4)

        x5 = self.fpa(x5)  # 1/32
        x4 = self.gau3(x4, x5)  # 1/16
        x3 = self.gau2(x3, x4)  # 1/8
        x2 = self.gau1(x2, x3)  # 1/4

        out = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(x2)

        return out


if __name__ == '__main__':

    import os
    import time

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    dummy_in = torch.randn(2, 3, 256, 256).cuda().requires_grad_()

    model = PAN(backbone='resnet34', pretrained=True, n_class=2)

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count())).cuda()

    while True:
        model.train()
        start_time = time.time()
        dummy_out = model(dummy_in)
        end_time = time.time()
        print("Inference time: {}s".format(end_time - start_time))