import logging
import math

import numpy as np
from PIL import Image
from paddle.vision import datasets


class CIFAR10SSL(datasets.Cifar10):
    def __init__(self, data_file, indexs, mode='train',
                 transform=None, download=True, backend='pil'):
        super().__init__(data_file=data_file, mode=mode,
                         transform=transform, download=download, backend=backend)
        if indexs is not None:
            # print(f"indexs: {indexs}")
            self.data = np.asarray(self.data)[indexs]
            # self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        image, label = self.data[index]
        # print(f"index {index}, image: {image.shape}")
        image = np.reshape(image, [3, 32, 32])
        image = image.transpose([1, 2, 0])  # HWC

        if self.backend == 'pil':
            image = Image.fromarray(image.astype('uint8'))
        if self.transform is not None:
            image = self.transform(image)

        if self.backend == 'pil':
            return image, np.array(label).astype('int64')

        return image, label
