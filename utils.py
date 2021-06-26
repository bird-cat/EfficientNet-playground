#import apex.amp as amp
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import os
import cv2
from randaugment import RandAugment, ImageNetPolicy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

class ImgDataset:
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        # Label should be the long tensor type
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X
            
def readfile(path, label):
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        if i % 100 == 0:
            print(i)
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(128, 128))
        if label:
          y[i] = int(file.split("_")[0])
    if label:
      return x, y
    else:
      return x

def get_loaders(data_directory, batch_size, image_size, image_crop):
    print('==> Preparing Food-11 dataset..')
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size,image_size)),
        transforms.RandomResizedCrop(image_crop),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size,image_size)),
        transforms.CenterCrop(image_crop),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    print("Reading data")
    train_x, train_y = readfile(data_directory+'/training', True)
    print("Size of training data = {}".format(len(train_x)))
    val_x, val_y = readfile(data_directory+'/validation', True)
    print("Size of validation data = {}".format(len(val_x)))
    test_x, test_y = readfile(data_directory+'/evaluation', True)
    print("Size of Testing data = {}".format(len(test_x)))
        
    train_dataset = ImgDataset(train_x, train_y, train_transform)
    test_dataset = ImgDataset(test_x, test_y, test_transform)
    val_dataset = ImgDataset(val_x, val_y, test_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,\
        shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,\
        shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,\
        shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    return train_loader, test_loader, val_loader

