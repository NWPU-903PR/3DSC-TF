from PIL import Image
import torch
import nibabel as nib
from skimage import transform
import numpy as np
import cv2
import numpy as np
from torch.utils.data import Dataset
from patchify import patchify
from utils import patches_proposal
from nibabel.viewers import OrthoSlicer3D



class MyDataSet_3D_60_patches(Dataset):

    def __init__(self, images_path: list, images_class: list, patches_loc,patch_size,patch_num,transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
        self.patches_loc = patches_loc
        self.p_size = patch_size
        self.p_num = patch_num

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = nib.load(self.images_path[item]).get_data()
        img = img[45:145, 56:181, 40:140]
        img = patchify(img,patch_size=(self.p_size,self.p_size,self.p_size),step=self.p_size//2)
        img_patch = np.zeros((self.p_num,self.p_size,self.p_size,self.p_size))
        patches_loc = self.patches_loc
        for i in range(self.p_num):
            loc_a = patches_loc.iloc[i, 0]
            loc_b = patches_loc.iloc[i, 1]
            loc_c = patches_loc.iloc[i, 2]
            img_patch[i] = img[loc_a, loc_b, loc_c]

        img_patch = img_patch.reshape((self.p_num,1,self.p_size,self.p_size,self.p_size))

        img = torch.from_numpy(img_patch).float()
        label = self.images_class[item]

        return img, label

    @staticmethod
    def collate_fn(batch):

        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels




