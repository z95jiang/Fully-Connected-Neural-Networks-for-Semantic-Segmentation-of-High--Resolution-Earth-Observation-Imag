import argparse
import json
import os
import os.path as osp
import warnings
import copy
import numpy as np
import PIL.Image

from labelme import utils

def main():
    json_file = './data/jason'
    tgt = './data/label_train'

    list = os.listdir(json_file)
    for i in range(0, len(list)):
        path = os.path.join(json_file, list[i])
        filename = list[i][:-5]       # .json
        if os.path.isfile(path):
            data = json.load(open(path,encoding='gbk'))
            # print(len(data['shapes']))
            # if len(data['shapes'])==1:
            #     print(len(data['shapes']))
            #     print(data['shapes'])
            img = utils.img_b64_to_arr(data['imageData'])
            # if len(data['shapes'])<4:
            #     print(len(data['shapes']))
            #     print(filename)
            #     continue
            lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes'])  # labelme_shapes_to_label

            captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
            lbl_viz = utils.draw_label(lbl, img, captions)
            out_dir = osp.basename(list[i]).replace('.', '_')
            out_dir = osp.join(osp.dirname(list[i]), out_dir)
            # if not osp.exists(out_dir):
            #     os.mkdir(out_dir)

            PIL.Image.fromarray(img).save(osp.join(tgt+'_vis', '{}_img.png'.format(filename)))
            PIL.Image.fromarray(lbl).save(osp.join(tgt, '{}.png'.format(filename)))
            PIL.Image.fromarray(lbl_viz).save(osp.join(tgt+'_vis', '{}_viz.png'.format(filename)))

            # with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
            #     for lbl_name in lbl_names:
            #         f.write(lbl_name + '\n')

            warnings.warn('info.yaml is being replaced by label_names.txt')
            info = dict(label_names=lbl_names)
            # with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
            #     yaml.safe_dump(info, f, default_flow_style=False)

            # print('Saved to: %s' % out_dir)


if __name__ == '__main__':
    main()
