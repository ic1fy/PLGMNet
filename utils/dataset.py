from torch.utils.data import Dataset
from torchvision import transforms
from utils.joint_transforms import *
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class my_dataset(Dataset):
    def __init__(self, file_path, in_size=384, is_train=True):
        self.ground_truth_list = []
        self.image_list = []
        self.is_train = is_train

        with open(file_path, 'r') as f:
            for data in f:
                data = data.split(" ")
                image_path, label_path = data[0], data[1]
                self.image_list.append(image_path)
                self.ground_truth_list.append(label_path)

        self.mask_transform = transforms.ToTensor()
        if self.is_train:
            self.joint_transform = Compose(
                [JointResize(in_size), RandomHorizontallyFlip(), RandomRotate(10)]
            )

            img_transform = [transforms.ColorJitter(0.1, 0.1, 0.1)]
            self.img_transform = transforms.Compose(
                [
                    *img_transform,
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.img_transform = transforms.Compose(
                [
                    transforms.Resize((in_size, in_size), interpolation=Image.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )


    def __len__(self):
        return len(self.ground_truth_list)

    def __getitem__(self, item):
        sample = self.transform(self.image_list[item], self.ground_truth_list[item])
        return sample

    def transform(self, image_path, mask_path):
        name = mask_path.split('/')[-1].split('.')[0]
        img = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.is_train:
            img, mask = self.joint_transform(img, mask)
        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        mask = mask.ge(0.5).float()

        sample = {'image':img, 'label':mask, 'name':name}
		
        return sample





	
    

