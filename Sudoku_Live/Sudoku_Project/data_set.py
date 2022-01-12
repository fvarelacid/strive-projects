import cv2 as cv
import numpy as np
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self):
        self.imgs_path = "74KFonts/"
        file_list = glob.glob(self.imgs_path + "*")
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for img_path in glob.glob(class_path + "/*.png"):
                self.data.append([img_path, class_name])
        self.class_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7, "9": 8}
        self.img_dim = (28, 28)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv.imread(img_path, cv.COLOR_BGR2GRAY)
        img = cv.resize(img, self.img_dim)
        class_id = self.class_map[class_name]
        img_tensor = self.transform(img)
        return img_tensor, class_id
