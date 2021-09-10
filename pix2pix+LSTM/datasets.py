import numpy as np
import os
from glob import glob
#import scipy
import imageio
import random

class DataLoader():
    def __init__(self, dataset_name, height=256, width=256):
        self.dataset_name = dataset_name
        self.height = height
        self.width = width

    def imread(self, path):
#        return scipy.misc.imread(path, mode='RGB').astype(np.float)
        image = imageio.imread(path, as_gray=True, pilmode='RGB').astype(np.float)
        return image.reshape(self.height, self.width, -1)

    def get_img_list(self):
        filenames = glob('data/{}/{}/{}/*.jpg'.format(self.dataset_name, 'hig_reso', '00'))
        str_len = len('data/{}/{}/{}/'.format(self.dataset_name, 'hig_reso', '00'))
        img_list = [filename[str_len:] for filename in filenames]
        img_list = list(sorted(img_list))
#        print(img_list)
        return img_list

    def make_train_val_set(self, train_size=0.8):
        dirs = [dir_name for dir_name in os.listdir('data/{}/{}/'.format(self.dataset_name, 'hig_reso'))]
#        print(dirs)

        train_dirs = random.sample(dirs, int(len(dirs)*train_size))
        val_dirs = list(set(dirs) - set(train_dirs))
        print('train_dirs= ', train_dirs)
        print('val_dirs= ', val_dirs)

        return train_dirs, val_dirs

    def load_batch(self, dir, img_list):
        imgs_A, imgs_B = [], []
        for img in img_list:
            filename_A = 'data/{}/{}/{}/{}'.format(self.dataset_name, 'low_reso', str(dir), img)
            filename_B = 'data/{}/{}/{}/{}'.format(self.dataset_name, 'hig_reso', str(dir), img)
            img_A = self.imread(filename_A)
            img_B = self.imread(filename_B)
            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.
        return imgs_A, imgs_B
#        yield imgs_A, imgs_B

    def get_img_shape(self, img):
        filename = 'data/{}/{}/{}/{}'.format(self.dataset_name, 'hig_reso', '00', img)
        img = self.imread(filename)
        return img.shape
