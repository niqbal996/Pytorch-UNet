import matplotlib.pyplot as plt
import cv2
import numpy as np


def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    # fig, ax = plt.subplots(1, classes + 1)
    # ax[0].set_title('Input image')
    # ax[0].imshow(img)
    mask = np.asarray(mask, dtype=np.uint8)

    # for i in range(classes):
        # ax[i+1].set_title(f'Output mask (class {i+1})')
    mask[0, :, :] = mask[0, :, :] * 0       # ax[i+1].imshow(mask[i, :, :])
    mask[1, :, :] = mask[1, :, :] * 128       # ax[i+1].imshow(mask[i, :, :])
    mask[2, :, :] = mask[2, :, :] * 255       # ax[i+1].imshow(mask[i, :, :])

    # mask = cv2.resize(mask, (img.size[0], img.size[1]), interpolation=cv2.INTER_LINEAR)
    # plt.xticks([]), plt.yticks([])
    # plt.show()
    mask = np.transpose(mask, [1, 2, 0])
    cv2.imshow('mask', mask)
    cv2.waitKey()