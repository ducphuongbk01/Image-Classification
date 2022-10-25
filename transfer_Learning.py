######### Download data ###########

'''
import os
import zipfile
import urllib.request

data_dir = "./data"

url = "http://download.pytorch.org/tutorial/hymenoptera_data.zip"
save_path = os.path.join(data_dir, "hymenoptera_data.zip")

if not os.path.exists(save_path):
    urllib.request.urlretrieve(url, save_path)

    # read by zip file
    zip = zipfile.ZipFile(save_path)
    zip.extractall(data_dir)
    zip.close()

    os.remove(save_path)
'''
import glob
import os.path as osp
import random
import numpy as np
import json 
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms
from tqdm import tqdm

torch.manual_seed(1234)
np,random.seed(1234)
random.seed(1234)

#torch.backends.cudnn.deterministic  = True
#torch.backends.cudnn.benchmark = False

# Tien xu ly anh truoc khi dua vao model
class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale = (0.5, 1, 0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),

            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
    
    def __call__(self, img, phase = 'train'):
        return self.data_transform[phase](img)

##img_file_path = './1.jpg'
##img = Image.open(img_file_path)

##plt.imshow(img)
##plt.show()

resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

##transform = ImageTransform(resize, mean, std)
##img_transformed = transform(img, phase = 'train')

#(channels, height, width) -> (height, width, channels)
##img_transformed = img_transformed.numpy().transpose(1,2,0)
##img_transformed = np.clip(img_transformed, 0, 1)
##plt.imshow(img_transformed)
##plt.show()

# Tao 1 list luu cac dia chi cua anh
def make_datapath_list(phase = 'train'):
    rootpath = "./data/hymenoptera_data/"
    target_path = osp.join(rootpath+phase+"/**/*.jpg")

    path_list = []

    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list

#path_list = make_datapath_list('train')
#print(path_list[0][30:34])

train_list = make_datapath_list('train')
val_list = make_datapath_list('val')

# Tao Dataset
class MyDataset(data.Dataset):
    def __init__(self, file_list, transform = None, phase = 'train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx): #Ham return output de dua vao model 
        img_path = self.file_list[idx]
        img = Image.open(img_path)

        img_transformed = self.transform(img, self.phase)

        # Lay label cua tam anh
        if self.phase == 'train':
            label = img_path[30:34]
        elif self.phase == 'val':
            label = img_path[28:32]
        
        if label == 'ants':
            label = 0
        elif label == 'bees':
            label = 1

        return img_transformed, label

train_dataset = MyDataset(train_list, transform=ImageTransform(resize, mean, std), phase = 'train')
val_dataset = MyDataset(val_list, transform=ImageTransform(resize, mean, std), phase = 'val')

#index = 200
#print(train_dataset.__len__())
#img, label = train_dataset.__getitem__(index)
#print(img.shape)
#print(label)

# Tao Dataloader
batch_size = 4

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle = True)  #shuffle giup xao tron data cho moi epoch

val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle = False)

dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}

##batch_iterator = iter(dataloader_dict['train'])
##inputs, labels = next(batch_iterator)

##print(inputs.shape)
##print(labels)

# Tao network, loss fnc, optimizer, training
use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)

net.classifier[6] = nn.Linear(in_features = 4096, out_features = 2)

# Setting model 
net.train()

# Loss function 
criterior = nn.CrossEntropyLoss()

# Optimizer
params_to_update = []

update_params_name = ["classifier.6.weight", "classifier.6.bias"]

for name, param in net.named_parameters():
    if name in update_params_name:
        param.requires_grad = True
        params_to_update.append(param)
    else:
        param.requires_grad = False


optimizer = optim.SGD(params = params_to_update, lr = 0.001, momentum = 0.9)

# Training
def train_model(net, dataloader_dict, criterior, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            if (epoch == 0) and (phase == 'train'):
                continue
            for inputs, labels in tqdm(dataloader_dict[phase]):
                optimizer.zero_grad() #reset cac gia tri dao ham cua cac epoch truoc do 

                with torch.set_grad_enabled(phase == 'train'): #neu la mode training thi enable grad cua tensor 
                    outputs = net(inputs)
                    loss = criterior(outputs, labels)
                    _, preds = torch.max(outputs,1)

                    if phase == 'train': 
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item()*inputs.size(0)
                    epoch_corrects += torch.sum(preds==labels.data)

            epoch_loss = epoch_loss/len(dataloader_dict[phase].dataset)
            epoch_accuracy = epoch_corrects.double()/len(dataloader_dict[phase].dataset)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_accuracy))

num_epochs = 2
train_model(net, dataloader_dict, criterior, optimizer, num_epochs)
