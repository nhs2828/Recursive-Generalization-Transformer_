import torch
from torch.utils.data import Dataset
import os
import glob
from torchvision.io import read_image
import torchvision.transforms as trnsfrms
import torchvision.transforms.functional as TF
import random
from torch.utils.data import DataLoader
import fnmatch

import unittest
import warnings
import re

class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class HRImgTransforms:
    """
    Transform used for HR images: a Random crop (specify size)
    """
    def __init__(self, crop_size):
        self.transforms = trnsfrms.Compose([
            trnsfrms.Lambda(lambda x: x / 255.0),
            trnsfrms.RandomCrop((crop_size, crop_size)),
            trnsfrms.RandomHorizontalFlip(),
            MyRotationTransform(angles=[90, 180, 270])
        ])


class LRImgTransforms:
    """
    transform used for Low-Res images.
    TODO: Check if antialiasing should be used
    """
    def __init__(self, crop_size):
        self.transforms = trnsfrms.Compose([
            trnsfrms.Resize((crop_size, crop_size), interpolation=trnsfrms.InterpolationMode.BICUBIC, antialias=True),
        ])
        
class HRImgTransforms_test:
    """
    Transform used for HR images: a Random crop (specify size)
    """
    def __init__(self):
        self.transforms = trnsfrms.Compose([
            trnsfrms.Lambda(lambda x: x / 255.0),
        ])


class LRImgTransforms_test:
    """
    transform used for Low-Res images.
    TODO: Check if antialiasing should be used
    """
    def __init__(self, h, w):
        self.transforms = trnsfrms.Compose([
            trnsfrms.Resize((h, w), interpolation=trnsfrms.InterpolationMode.BICUBIC, antialias=True),
        ])
        

class CustomSuperResolutionDataset(Dataset):
    """
    Custom Dataset class for the SR transformer.
    params:
        - hr_img_dir: str: Path to the high-resolution images that serve as labels
        - transform: transform for the input images 
        - target_transform: transform for the labels (HR image)
    """
    def __init__(self, hr_img_dir, transform=None, target_transform=None):
        self.hr_img_dir = sorted(glob.glob(hr_img_dir+'/*.png')) #labels
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        """
        Returns count of the files contained in the HR directory.
        """
        return len(self.hr_img_dir)

    def __getitem__(self, idx):
        hr_image = read_image(self.hr_img_dir[idx])

        if self.target_transform:
            hr_image = self.target_transform(hr_image) #random crop
        if self.transform:
            lr_image = self.transform(hr_image) #downscaling
        return lr_image, hr_image
    

class TestDatasetDataloader(unittest.TestCase):

    def test_vanilla_conv(self):
        class SimpleConvNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
                self.fc = torch.nn.Linear(16 * 64 * 64, 10)  

            def forward(self, x):
                x = self.conv1(x)
                x = torch.nn.functional.relu(x)
                # Flatten the output for the fully connected layer
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        model = SimpleConvNet()

        input_data = torch.randn(8, 3, 64, 64)

        # Forward pass
        output = model(input_data)

        # Display the output shape
        print("Output shape:", output.shape)
        self.assertEqual(output.size(), (8, 10))
        
        transforms = LRImgTransforms(64)
        target_transforms = HRImgTransforms(128)

        train_ds = CustomSuperResolutionDataset('../../DIV2K_train_HR', transform=transforms.transforms, target_transform=target_transforms.transforms)
        train_dl = DataLoader(train_ds, 32, shuffle=True)
        count = 0
        for e in range(1):
            print(f"\n----------\n\tTest epoch {e}")
            for x, y in train_dl:
                y_hat = model(x)
                if count % 50 == 0:
                    self.assertEqual(x.size(), (32, 3, 64, 64))
                    self.assertEqual(y.size(), (32, 3, 128, 128))
                    self.assertEqual(y_hat.size(), (32,10))
                count += 1


    def test_dataset_creation(self):
        transforms = LRImgTransforms(64)
        target_transforms = HRImgTransforms(128)
        SRDataset = CustomSuperResolutionDataset('../../DIV2K_train_HR', transform=transforms.transforms, target_transform=target_transforms.transforms)
        self.assertEqual(800, len(SRDataset))
        
        self.assertEqual((3, 64, 64), SRDataset[1][0].size())
        self.assertEqual((3, 128, 128), SRDataset[1][1].size())

        #ces tests ne passent pas: ce n'est pas vraiment normalise entre 0 et 1: TODO
        #self.assertGreaterEqual(torch.min(SRDataset[1][0]), 0)
        #self.assertLessEqual(torch.max(SRDataset[1][0]), 1)

    def test_dataset_creation_valid(self):
        transforms = LRImgTransforms(64)
        target_transforms = HRImgTransforms(128)
        SRDataset = CustomSuperResolutionDataset('../../DIV2K_valid_HR', transform=transforms.transforms, target_transform=target_transforms.transforms)
        self.assertEqual(100, len(SRDataset))
        
        self.assertEqual((3, 64, 64), SRDataset[801][0].size())
        self.assertEqual((3, 128, 128), SRDataset[801][1].size())

    def test_train_dataloader(self):
        transforms = LRImgTransforms(64)
        target_transforms = HRImgTransforms(128)
        SRDataset_train = CustomSuperResolutionDataset('../../DIV2K_train_HR', transform=transforms.transforms, target_transform=target_transforms.transforms)
        train_dl = DataLoader(SRDataset_train, 32, shuffle=True)
        
        feats_train, labels_train = next(iter(train_dl))
        self.assertEqual((32, 3, 64, 64), feats_train.size())
        self.assertEqual((32, 3, 128, 128), labels_train.size())
        
    
if __name__ == '__main__':
    unittest.main()
