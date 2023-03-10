from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """

    def __init__(self,
                 base_dir=None,
                 split='train',
                 transform=None
                 ):

        self._base_dir = base_dir
        self.split = split
        self._image_dir = os.path.join(self._base_dir, 'image_'+self.split)
        self._cat_dir = os.path.join(self._base_dir, 'label_'+self.split)
        self.categories = os.listdir(self._cat_dir)
        self.images = []
        for i in range(len(self.categories)):
            self.images.append(self.categories[i])
        self.transform = transform


        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img, _target= self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'gt': _target}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _make_img_gt_point_pair(self, index):
        # Read Image and Target
        _img = np.asarray(Image.open(os.path.join(self._image_dir, self.images[index].replace('png', 'jpg'))).convert('RGB')).astype(np.float32)
        # _target = np.asarray(Image.open(os.path.join(self._cat_dir, self.categories[index])).convert('L')).astype(np.int32)
        # _img = io.read_image(os.path.join(self._image_dir, self.images[index]), driver = 'GDAL')
        _target = np.asarray(Image.open(os.path.join(self._cat_dir, self.categories[index])).convert('L')).astype(
            np.int32)
        return _img, _target

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'

