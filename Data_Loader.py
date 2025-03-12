from __future__ import print_function, division
import os
from PIL import Image
import torch
import torch.utils.data
import torchvision
from torch.utils.data import Dataset
import random
import numpy as np
from utils import tomasks, readColormap

class Images_Dataset_folder(torch.utils.data.Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        transformI = Input Images transformation (default: None)
        transformM = Input Labels transformation (default: None)
    Output:
        tx = Transformed images
        lx = Transformed labels, note that the labels are converted to masks instead of RGB values
        and the lables have been converted to one hot encoding"""

    def __init__(self, images_dir, labels_dir,transformI = None, transformM = None, num_classes=12):
        self.images = sorted(os.listdir(images_dir))
        self.labels = sorted(os.listdir(labels_dir))
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transformI = transformI
        self.transformM = transformM
        self.num_classes = num_classes

        if self.transformI:
            self.tx = self.transformI
        else:
            self.tx = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])

        if self.transformM:
            self.lx = self.transformM
        else:
            self.lx = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])

    def __len__(self):

        return len(self.images)

    def __getitem__(self, i):
        i1_path = os.path.join(self.images_dir, self.images[i])
        l1_path = os.path.join(self.labels_dir, self.labels[i])
        i1 = Image.open(i1_path)
        l1 = Image.open(l1_path)
        seed = np.random.randint(0, 2**31 - 1)

        # apply this seed to img tranfsorms
        random.seed(seed) 
        torch.manual_seed(seed)
        img = self.tx(i1)
        label = np.array(l1, dtype=np.uint8)
        label = torch.tensor(label, dtype=torch.long)
        # apply this seed to target/label tranfsorms  
        random.seed(seed) 
        torch.manual_seed(seed)
    
        # Convert to one hot encoding
        label = torch.nn.functional.one_hot(label.long(), num_classes= self.num_classes).permute(2, 0, 1).float()
        return img, label



