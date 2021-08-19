from os.path import splitext
from os import listdir

import cv2
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess_img(cls, pil_img, scale):
        # w, h = pil_img.size
        # newW, newH = int(scale * w), int(scale * h)
        # assert newW > 0 and newH > 0, 'Scale is too small'
        # pil_img = pil_img.resize((newW, newH))

        # img_nd = np.array(pil_img)
        cv_image = np.array(pil_img)
        img_nd = cv2.resize(cv_image, (256, 256), interpolation=cv2.INTER_NEAREST_EXACT)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    @classmethod
    def preprocess_mask(cls, pil_img):
        cv_image = np.array(pil_img)
        img_nd = cv2.resize(cv_image, (256, 256), interpolation=cv2.INTER_NEAREST_EXACT)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        # classes = [0, 128, 255]
        # masks = np.zeros([3, img_trans.shape[1], img_trans.shape[2]])
        # for class_id, channel in zip(classes, range(len(classes))):
        #     mask_tmp = masks[channel, :, :].flatten()
        #     orig = img_trans[0, :, :].flatten()
        #     mask_tmp[np.where(orig == class_id)] = orig[np.where(orig == class_id)]
        #     masks[channel, :, :] = np.reshape(mask_tmp, (img_trans.shape[1], img_trans.shape[2]))
        #     if class_id == 0:
        #         masks[channel, :, :] = np.reshape(mask_tmp, (img_trans.shape[1], img_trans.shape[2]))
        #     else:
        #         masks[channel, :, :] = np.reshape(mask_tmp, (img_trans.shape[1], img_trans.shape[2]))
        #         masks[channel, :, :] = masks[channel, :, :] / class_id
        img_trans = np.round(img_trans / 128)

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess_img(img, self.scale)
        mask = self.preprocess_mask(mask)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
