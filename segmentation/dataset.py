import csv
import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data

from torchvision import transforms
from torchvision.utils import make_grid

import torch
from sklearn.preprocessing import OneHotEncoder


####
class DatasetSerial(data.Dataset):
    @staticmethod
    def _isimage(image, ends):
        return any(image.endswith(end) for end in ends)

    def __init__(self, pair_list, shape_augs=None, input_augs=None):
        self.pair_list = pair_list
        self.shape_augs = shape_augs
        self.input_augs = input_augs

    def __getitem__(self, idx):
        pair = self.pair_list[idx]


        input_img = cv2.imread(pair[0])
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        # shape must be deterministic so it can be reused
        shape_augs = self.shape_augs.to_deterministic()
        input_img = shape_augs.augment_image(input_img)

        img_label = cv2.imread(pair[1])
        img_label = shape_augs.augment_image(img_label)
        # actually pseudo label
        img_label[img_label > 0] = 1
        img_label = img_label[...,0]

        # additional augmentation just for the input
        if self.input_augs is not None:
            input_img = self.input_augs.augment_image(input_img)

        img_label=np.copy(img_label)
        return input_img, img_label

    def __len__(self):
        return len(self.pair_list)


####

####
def prepare_data_viable():
    train_pairs = []
    valid_pairs = []

    root_dir = 'Segmentation_viable/'
    original_dir = 'Segmentation_viable/original_image/'
    label_dir = root_dir + 'viable_label_image/'

    for i in range(1, 21):
        original_list = glob.glob(original_dir + 'Training_phase_1_{0:03d}'.format(i) + '/*.jpg')
        label_list = glob.glob(label_dir + 'Training_phase_1_{0:03d}'.format(i) + '/*.jpg')
        train_pairs += list(zip(original_list, label_list))

    for i in range(21, 41):
        original_list = glob.glob(original_dir + 'Training_phase_2_{0:03d}'.format(i) + '/*.jpg')
        label_list = glob.glob(label_dir + 'Training_phase_2_{0:03d}'.format(i) + '/*.jpg')
        train_pairs += list(zip(original_list, label_list))


    for i in range(41, 51):
        original_list = glob.glob(original_dir + 'Training_phase_2_{0:03d}'.format(i) + '/*.jpg')
        label_list = glob.glob(label_dir + 'Training_phase_2_{0:03d}'.format(i) + '/*.jpg')
        valid_pairs += list(zip(original_list, label_list))

    return train_pairs, valid_pairs

####
def prepare_data_CANCER():
    train_pairs = []
    valid_pairs = []

    root_dir ='/media/disk/han/dataset/Segmentation_whole/'
    original_dir = root_dir+'original_image/'
    label_dir = '/media/disk/han/dataset/Segmentation_whole/whole_label_image/'

    for i in range(1, 51):
        original_list = glob.glob(original_dir + '{0:03d}'.format(i) + '/*.png')
        label_list = glob.glob(label_dir + '{0:03d}'.format(i) + '/*.png')
        train_pairs += list(zip(original_list, label_list))


    np.random.shuffle(train_pairs)

    print(len(train_pairs))


    return train_pairs[:-1], train_pairs[-1]
####
def prepare_data_VIABLE_2048():
    train_pairs = []
    valid_pairs = []


    original_dir = '/media/disk/han/dataset/traindataset_2048/original_img/'
    label_dir = '/media/disk/han/dataset/traindataset_2048/label/'

    for i in range(1, 41):
        original_list = glob.glob(original_dir + '{0:03d}'.format(i) + '/*.png')
        label_list = glob.glob(label_dir + '{0:03d}'.format(i) + '/*.png')
        train_pairs += list(zip(original_list, label_list))

    for i in range(41, 51):
        original_list = glob.glob(original_dir + '{0:03d}'.format(i) + '/*.png')
        label_list = glob.glob(label_dir + '{0:03d}'.format(i) + '/*.png')
        valid_pairs += list(zip(original_list, label_list))

    np.random.shuffle(train_pairs)
    np.random.shuffle(valid_pairs)
    print(len(train_pairs))
    print(len(valid_pairs))

    return train_pairs[:], valid_pairs[:3000]


def prepare_data_test():
    train_pairs = []
    valid_pairs = []

    root_dir = 'test/'
    original_dir = 'test/original_image/'
    label_dir = root_dir + 'viable_label_image/'

    original_list = glob.glob(original_dir + 'train' + '/*.jpg')
    label_list = glob.glob(label_dir + 'train' + '/*.jpg')
    train_pairs += list(zip(original_list, label_list))

    original_list = glob.glob('test/original_image/val' + '/*.jpg')
    label_list = glob.glob('test/whole_label_image/val'+'/*.jpg')
    valid_pairs += list(zip(original_list, label_list))

    return train_pairs, valid_pairs
####
def visualize(ds, batch_size, nr_steps=100):
    data_idx = 0
    cmap = plt.get_cmap('jet')
    for i in range(0, nr_steps):
        if data_idx >= len(ds):
            data_idx = 0
        for j in range(1, 2 * batch_size + 1, 2):
            sample = ds[data_idx + j - 1]
            plt.subplot(2, batch_size, j)
            plt.imshow(np.squeeze(sample[0]))
            plt.subplot(2, batch_size, j + 1)
            plt.imshow(np.squeeze(sample[1]))

        plt.show()
        data_idx += batch_size
