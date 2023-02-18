batch_size = 2
use_gpu = True
mean = (0.315, 0.319, 0.470)
std = (0.144, 0.151, 0.211)
img_size = 256
epoches = 4
base_lr = 0.0001
weight_decay = 2e-5
momentum = 0.9
power = 0.99

num_class = 2  # some parameters
model_name = 'segnet'  #'PAN, pspnet , segnet, refinenet, unet1, UNet, UNet_2Plus, UNet_3Plus'
input_bands = 3
data_dir='./data'