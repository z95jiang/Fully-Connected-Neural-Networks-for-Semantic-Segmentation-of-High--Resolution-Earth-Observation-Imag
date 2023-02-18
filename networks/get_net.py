from networks import *


def get_net(model_name, input_bands, num_class, img_size):
    if model_name == 'pspnet':
        net = PSPNet(input_bands, n_classes=num_class, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024,
                     backend='resnet50', pretrained=False)
    elif model_name == 'densenet_aspp':
        net = densenet_aspp(input_bands, num_class)
    elif model_name == 'segnet':
        net = segnet(input_bands, num_class)
    elif model_name == 'refinenet':
        net = RefineNet4Cascade(input_shape=(input_bands, img_size, img_size), num_classes=num_class)
    elif model_name == 'unet1':
        net = unet(input_bands=input_bands, n_classes=num_class)
    elif model_name == 'UNet':
        net = UNet(in_channels=input_bands, n_classes=num_class)
    elif model_name == 'UNet_2Plus':
        net = UNet_2Plus(in_channels=input_bands, n_classes=num_class)
    elif model_name == 'UNet_3Plus':
        net = UNet_3Plus(in_channels=input_bands, n_classes=num_class)
    elif model_name == 'DeepLabv3_plus':
        net = DeepLabv3_plus(num_class)
    elif model_name == 'BiSeNet':
        net = BiSeNet(n_classes=num_class, context_path='resnet18')
    elif model_name == 'PAN':
        net = PAN(backbone='resnet34', pretrained=True, n_class=num_class)
    else:
        raise (
            'this model is not exist!!!!, existing mode is pspnet ,densenet_aspp, segnet, refinenet, unet1, UNet, UNet_2Plus, UNet_3Plus')
    return net
