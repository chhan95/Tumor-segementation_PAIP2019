import csv
import glob
import random
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms
from torchvision.utils import make_grid
from imgaug import augmenters as iaa

import numpy as np


####
class DatasetSerial(data.Dataset):

    def __init__(self, pair_list, shape_augs=None, input_augs=None):
        self.pair_list = pair_list
        self.shape_augs = iaa.Sequential(shape_augs)
        self.input_augs = iaa.Sequential(input_augs)

    def __getitem__(self, idx):

        pair = self.pair_list[idx]

        input_img = cv2.imread(pair[0])
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        img_label = pair[1]  # normal is 0

        # shape must be deterministic so it can be reused
        if self.shape_augs is not None:
            shape_augs = self.shape_augs.to_deterministic()
            input_img = shape_augs.augment_image(input_img)

        if self.input_augs is not None:
            input_img = iaa.Sequential([
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),  # gaussian blur with random sigma
                    iaa.MedianBlur(k=(3, 5)),  # median with random kernel sizes
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                ]),

                iaa.Add((-26, 26)),
                iaa.AddToHueAndSaturation((-10, 10)),
                iaa.LinearContrast((0.8, 1.2), per_channel=1.0),
            ], random_order=True).augment_image(input_img)

        return input_img, img_label

    def __len__(self):
        return len(self.pair_list)



def prepare_PAIP_data():
    def load_data_info(pathname, parse_label=True, label_value=0):
        file_list = glob.glob(pathname)
        if parse_label:
            label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
        else:
            label_list = [label_value for file_path in file_list]
        print(pathname)
        print(Counter(label_list))
        return list(zip(file_list, label_list))

   # data_root_dir = '/media/exhard_1/2019_2/PAIP2019/dataset/for_train/patch_classification/for_tumor'
    data_root_dir='/media/PAIP2020/training_set/'
    train_set = []
    valid_set = []
    for i in range(1, 41):
        train_set += load_data_info('%s/not_tumor/%s/*.png' % (data_root_dir, i),
                                    parse_label=False, label_value=0)
    for i in range(1, 41):
        train_set += load_data_info('%s/tumor/%s/*.png' % (data_root_dir, i),
                                    parse_label=False, label_value=1)

    for i in range(41, 51):
        valid_set += load_data_info('%s/not_tumor/%s/*.png' % (data_root_dir, i),
                                    parse_label=False, label_value=0)
    for i in range(41, 51):
        valid_set += load_data_info('%s/tumor/%s/*.png' % (data_root_dir, i),
                                    parse_label=False, label_value=1)

    np.random.shuffle(train_set)

    return train_set[:], valid_set[:]


####
def visualize(ds, batch_size, nr_steps=100):
    data_idx = 0
    cmap = plt.get_cmap('jet')
    for i in range(0, nr_steps):
        if data_idx >= len(ds):
            data_idx = 0
        for j in range(1, batch_size + 1):
            sample = ds[data_idx + j]
            if len(sample) == 2:
                img = sample[0]
            else:
                img = sample[0]
                # TODO: case with multiple channels
                aux = np.squeeze(sample[-1])
                aux = cmap(aux)[..., :3]  # gray to RGB heatmap
                aux = (aux * 255).astype('uint8')
                img = np.concatenate([img, aux], axis=0)
                img = cv2.resize(img, (40, 80), interpolation=cv2.INTER_CUBIC)
            plt.subplot(1, batch_size, j)
            plt.title(str(sample[1]))
            plt.imshow(img)
        plt.show()
        data_idx += batch_size
