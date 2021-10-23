import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets

logger = logging.getLogger(__name__)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        image, target = self.data[index], self.targets[index]
        image = np.reshape(image, [3, 32, 32])
        image = image.transpose([1, 2, 0])  # HWC

        image = Image.fromarray(image.astype('uint8'))

        if self.transform is not None:
            image = self.transform(image)


        return image, target
