from typing import List, Tuple

import numpy as np
import cv2
import random
from copy import deepcopy
import torch.utils.data


class CNNDataset(torch.utils.data.Dataset):

    def __init__(self, data, input_config, is_train):
        self.data = data
        self.input_config = input_config
        self.is_train = is_train
    def __getitem__(self, index):
        # noinspection PyArgumentList
        # mu, sigma = np.mean(self.data[index][0]), np.std(self.data[index][0])
        # input_arr = (self.data[index][0] - mu) / sigma
        x = self.data[index][0]
        # noinspection PyArgumentList
        
        output_arr = self.data[index][1]

        if self.input_config['augment_affine'] and self.is_train:
            # print(x.shape)
            # stack left & right channel together for augmentation
            x = x.squeeze()
            # cv2.imwrite('before_aug.png', x)
            rows, cols = x.shape
            cols = cols // 2
            x.shape = (rows, cols, 2)
            config_affine = self.input_config['affine_parameters']

            if random.random() > config_affine['scale'][2]:
                scale_x, scale_y = 1, 1
            else:
                scale_x = random.uniform(config_affine['scale'][0][0],
                                        config_affine['scale'][1][0])
                scale_y = random.uniform(config_affine['scale'][0][1],
                                        config_affine['scale'][1][1])
            x = cv2.resize(x, None, fx=scale_x, fy=scale_y)
            if random.random() > config_affine['rotate'][1]:
                rotate = 0
            else:
                rotate = random.uniform(-config_affine['rotate'][0],
                                        config_affine['rotate'][0])
            rows1, cols1, _ = x.shape
            M = cv2.getRotationMatrix2D(
                ((cols1 - 1) / 2.0, (rows1 - 1) / 2.0), rotate, 1)
            if random.random() > config_affine['move'][2]:
                movex = 0
                movey = 0
            else:
                movex = random.randrange(config_affine['move'][0][0],
                                        config_affine['move'][1][0])
                movey = random.randrange(config_affine['move'][0][1],
                                        config_affine['move'][1][1])
            M[0, 2] += movex + (cols - cols1) // 2
            M[1, 2] += movey + (rows - rows1) // 2
            x = cv2.warpAffine(x, M, (int(cols), int(rows)))

            x.shape = (rows, cols * 2)
            x = x[None, ...]
            # cv2.imwrite('after_aug.png', x)
            # input()
        
        return x, output_arr

    def __len__(self):
        return len(self.data)
