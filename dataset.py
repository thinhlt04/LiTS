import torch
from torch.utils.data import Dataset, DataLoader
import os
import SimpleITK as sitk
import numpy as np
import cv2


class LiTS(Dataset):
    def __init__(
        self,
        root,
        lowerbound,
        upperbound,
        train=False,
        dev=False,
        transform=None,
        target_transform=None,
    ):
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.images = []
        self.targets = []
        self.lowerbound = lowerbound
        self.upperbound = upperbound

        if train:
            root = os.path.join(root, "train")
        elif dev:
            root = os.path.join(root, "dev")
        else:
            root = os.path.join(root, "test")
        image_folder = os.path.join(root, "image")
        target_folder = os.path.join(root, "target")
        if self.liver_mask:
            self.liver_masks = []
            mask_folder = os.path.join(root, "liver_mask")
            for image, target, liver_mask in zip(
                sorted(os.listdir(image_folder)),
                sorted(os.listdir(target_folder)),
                sorted(os.listdir(mask_folder))):
                
                image_path = os.path.join(image_folder, image)
                target_path = os.path.join(target_folder, target)
                mask_path = os.path.join(mask_folder, liver_mask)
                self.images.append(image_path)
                self.targets.append(target_path)
                self.liver_masks.append(mask_path)
        else:
            for image, target in zip(
                    sorted(os.listdir(image_folder)), 
                    sorted(os.listdir(target_folder))):

                image_path = os.path.join(image_folder, image)
                target_path = os.path.join(target_folder, target)
                self.images.append(image_path)
                self.targets.append(target_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = sitk.ReadImage(self.images[idx])
        target = sitk.ReadImage(self.targets[idx])

        clamp_filter = sitk.ClampImageFilter()
        clamp_filter.SetLowerBound(self.lowerbound)
        clamp_filter.SetUpperBound(self.upperbound)
        clamped_image = clamp_filter.Execute(image)

        target = sitk.GetArrayFromImage(target)
        clamped_image = sitk.GetArrayFromImage(clamped_image)
        target[target == 2] = 1
        target = target.astype(np.float32)

        if self.transform:
            clamped_image = clamped_image.transpose(1, 2, 0)
            clamped_image = self.transform(clamped_image)
        if self.target_transform:
            target = target.transpose(1, 2, 0)
            target = self.target_transform(target)
        target = (target > 0).float() 
        return clamped_image.float(), target


class LiTS_stage2(Dataset):
    def __init__(
        self,
        root,
        lowerbound,
        upperbound,
        train=False,
        dev=False,
        transform=None,
        target_transform=None,
    ):
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.images = []
        self.targets = []
        self.liver_masks = []
        self.lowerbound = lowerbound
        self.upperbound = upperbound

        if train:
            root = os.path.join(root, "train")
        elif dev:
            root = os.path.join(root, "dev")
        else:
            root = os.path.join(root, "test")

        image_folder = os.path.join(root, "image")
        target_folder = os.path.join(root, "target")
        mask_folder = os.path.join(root, "liver_mask")
        
        for image, target, liver_mask in zip(
                sorted(os.listdir(image_folder)),
                sorted(os.listdir(target_folder)),
                sorted(os.listdir(mask_folder))):
            
            image_path = os.path.join(image_folder, image)
            target_path = os.path.join(target_folder, target)
            mask_path = os.path.join(mask_folder, liver_mask)

            self.images.append(image_path)
            self.targets.append(target_path)
            self.liver_masks.append(mask_path)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = sitk.ReadImage(self.images[idx])
        target = sitk.ReadImage(self.targets[idx])
        liver_mask = sitk.ReadImage(self.liver_masks[idx])

        clamp_filter = sitk.ClampImageFilter()
        clamp_filter.SetLowerBound(self.lowerbound)
        clamp_filter.SetUpperBound(self.upperbound)
        clamped_image = clamp_filter.Execute(image)
        

        
        masked_image = sitk.Mask(clamped_image, liver_mask, outsideValue=-1024)
        masked_target = sitk.Mask(target, liver_mask, outsideValue=0)

        liver_mask = sitk.GetArrayFromImage(liver_mask).astype(np.uint8)
        masked_target = sitk.GetArrayFromImage(masked_target).astype(np.uint8)
        masked_image = sitk.GetArrayFromImage(masked_image).astype(np.float32)
        target = sitk.GetArrayFromImage(target).astype(np.uint8)

        # img = masked_image[0]  

        # blur = cv2.GaussianBlur(img, (5,5), 0)
        # unsharp_mask = cv2.addWeighted(img, 1+3.0, blur, -3.0, 0)

        # sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        # sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        # sobel_mask = cv2.magnitude(sobel_x, sobel_y)

        # input = np.stack([img, unsharp_mask, sobel_mask], axis=0)

        masked_target[masked_target == 2] = 0
        masked_target = masked_target.astype(np.float32)
        target[target == 1] = 0
        target[target == 2] = 1

        if self.transform:
            masked_image = masked_image.transpose(1, 2, 0)
            masked_image = self.transform(masked_image)
        if self.target_transform:
            masked_target = masked_target.transpose(1, 2, 0)
            masked_target = self.target_transform(masked_target)
        masked_target = (masked_target > 0).float()

        return masked_image.float(), masked_target, liver_mask, target
