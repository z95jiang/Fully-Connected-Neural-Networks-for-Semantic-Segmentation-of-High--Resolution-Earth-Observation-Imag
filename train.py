from __future__ import division
import torch
import re
import time
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import tqdm
import os
import transform as tr
from dataloader import VOCSegmentation
from torchvision.utils import make_grid
import utils as utils
from tensorboardX import SummaryWriter
from pred_txt import test_my
from networks.get_net import get_net
from config import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_list = [0]


def main():
    composed_transforms_tr = standard_transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.ScaleNRotate(rots=(-15, 15), scales=(.75, 1.5)),
        # tr.RandomResizedCrop(img_size),
        tr.FixedResize(img_size),
        tr.Normalize(mean=mean, std=std),
        tr.ToTensor()])  # data pocessing and data augumentation

    voc_train = VOCSegmentation(base_dir=data_dir, split='train', transform=composed_transforms_tr)  # get data
    trainloader = DataLoader(voc_train, batch_size=batch_size, shuffle=True, num_workers=1)  # define traindata
    if use_gpu:
        model = torch.nn.DataParallel(frame_work, device_ids=gpu_list)  # use gpu to train
        model_id = 0
        if find_new_file(model_dir) is not None:
            model.load_state_dict(torch.load(find_new_file(model_dir)))
            # model.load_state_dict(torch.load('./pth/best2.pth'))
            print('load the model %s' % find_new_file(model_dir))
            model_id = re.findall(r'\d+', find_new_file(model_dir))
            model_id = int(model_id[0])
        model.cuda()
    else:
        # model = UNet(num_class)
        model = torch.nn.DataParallel(frame_work, device_ids=gpu_list)
        model_id = 0
        if find_new_file(model_dir) is not None:
            model.load_state_dict(torch.load(find_new_file(model_dir)))
            # model.load_state_dict(torch.load('./pth/best2.pth'))
            print('load the model %s' % find_new_file(model_dir))
            model_id = re.findall(r'\d+', find_new_file(model_dir))
            model_id = int(model_id[0])

    criterion = torch.nn.CrossEntropyLoss()  # define loss
    # optimizer = torch.optim.SGD(model.parameters(),lr=0.00001, momentum=momentum, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # define optimizer
    writer = SummaryWriter()
    model.train()

    f = open('log_{}.txt'.format(model_name), 'w')
    f.writelines('iou, miou, acc, prec, rec, f1_s\n')

    for epoch in range(epoches):
        cur_log = ''
        running_loss = 0.0
        start = time.time()
        lr = adjust_learning_rate(base_lr, optimizer, epoch, model_id, power)  # adjust learning rate
        for i, data in tqdm.tqdm(enumerate(trainloader)):  # get data

            images, labels = data['image'], data['gt']
            i += images.size()[0]
            labels = labels.view(images.size()[0], img_size, img_size).long()
            if use_gpu:
                images = Variable(images).cuda()
                labels = Variable(labels).cuda()
            else:
                images = Variable(images)
                labels = Variable(labels)
            optimizer.zero_grad()
            if model_name != 'pspnet':
                outputs = model(images)  # get prediction
            else:
                outputs, _ = model(images)
            losses = criterion(outputs, labels)  # calculate loss
            losses.backward()
            optimizer.step()
            running_loss += losses
            if i % 200 == 0:
                grid_image = make_grid(images[:3].clone().cpu().data, 3, normalize=True)
                writer.add_image('image', grid_image)
                grid_image = make_grid(
                    utils.decode_seg_map_sequence(torch.max(outputs[:3], 1)[1].detach().cpu().numpy()),
                    3,
                    normalize=False,
                    range=(0, 255))
                writer.add_image('Predicted label', grid_image)
                grid_image = make_grid(
                    utils.decode_seg_map_sequence(torch.squeeze(labels[:3], 1).detach().cpu().numpy()),
                    3,
                    normalize=False, range=(0, 255))
                writer.add_image('Groundtruth label', grid_image)

        print("Epoch [%d] all Loss: %.4f" % (epoch + 1 + model_id, running_loss / i))
        cur_log += 'epoch:{}, '.format(str(epoch)) + 'learning_rate:{}'.format(str(lr)) + ', ' + 'train_loss:{}'.format(
            str(running_loss.item() / i)) + ', '
        writer.add_scalar('learning_rate', lr, epoch + model_id)
        writer.add_scalar('train_loss', running_loss / i, epoch + model_id)

        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        torch.save(model.state_dict(), os.path.join(model_dir, '%d.pth' % (model_id + epoch + 1)))
        iou, miou, acc, prec, rec, f1_s = test_my(input_bands, model_name, model_dir, img_size, num_class)
        writer.add_scalar('iou', iou, epoch + model_id)
        writer.add_scalar('miou', miou, epoch + model_id)
        writer.add_scalar('iou', iou, epoch + model_id)
        writer.add_scalar('prec', prec, epoch + model_id)
        writer.add_scalar('rec', rec, epoch + model_id)
        writer.add_scalar('f1_s', f1_s, epoch + model_id)

        print([iou, miou, acc, prec, rec, f1_s])
        end = time.time()
        time_cha = end - start
        left_steps = epoches - epoch - model_id
        print('the left time is %d hours, and %d minutes' % (int(left_steps * time_cha) / 3600,
                                                             (int(left_steps * time_cha) % 3600) / 60))
        f.writelines('{}, {}, {}, {}, {}, {}\n'.format(iou, miou, acc, prec, rec, f1_s))


def find_new_file(dir):
    file_lists = os.listdir(dir)
    file_lists.sort(key=lambda fn: os.path.getmtime(dir + fn)
    if not os.path.isdir(dir + fn) else 0)
    if len(file_lists) != 0:
        file = os.path.join(dir, file_lists[-1])
        return file
    else:
        return None


def adjust_learning_rate(base_lr, optimizer, epoch, model_id, power):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = base_lr * ((1-float(epoch+model_id)/num_epochs)**power)
    lr = base_lr * (power ** ((epoch + model_id) // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    frame_work = get_net(model_name, input_bands, num_class, img_size)
    model_dir = './pth_{}/'.format(model_name)
    if os.path.exists(model_dir) is False:
        os.mkdir(model_dir)
    main()
