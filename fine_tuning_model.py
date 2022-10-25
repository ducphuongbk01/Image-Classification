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

resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

save_path = './weight_fine_tuning.pth'

# Tien xu ly anh
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


#Tao list data
def make_datapath_list(phase = 'train'):
    rootpath = "./data/hymenoptera_data/"
    target_path = osp.join(rootpath+phase+"/**/*.jpg")

    path_list = []

    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list

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

# Tao DataLoader
batch_size = 4
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle = True)  #shuffle giup xao tron data cho moi epoch
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle = False)
dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}

# Tao Network
use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)
net.classifier[6] = nn.Linear(in_features = 4096, out_features = 2)

#Tao loss fnc
criterior = nn.CrossEntropyLoss()

# Optmizer in fine tuning
def params_to_update(net):
    params_to_update1 = []
    params_to_update2 = []
    params_to_update3 = []

    update_param_name_1 = ["features"]
    update_param_name_2 = ["classifier.0.weight", "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]
    update_param_name_3 = ["classifier.6.weight", "classifier.6.bias"]

    for name, param in net.named_parameters():
        if name in update_param_name_1:
            param.requires_grad = True
            params_to_update1.append(param)
        elif name in update_param_name_2:
            param.requires_grad = True
            params_to_update2.append(param)
        elif name in update_param_name_3:
            param.requires_grad = True
            params_to_update3.append(param)
        else:
            param.requires_grad = False
    return  params_to_update1, params_to_update2, params_to_update3       


params1, params2, params3 = params_to_update(net)
optimizer = optim.SGD([
    {'params':params1, 'lr': 1e-4},
    {'params':params2, 'lr': 5e-4},
    {'params':params3, 'lr': 1e-3}
], momentum = 0.9)

# Ham training
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
    
    torch.save(net.state_dict(), save_path)


def load_model(net, model_path):
    load_weights = torch.load(model_path)
    net.load_state_dict(load_weights)

    # Neu Neu parameters luu tren GPU ma muon xai tren CPU thi lam cach nhu sau
    ##load.weights = torch.load(model_path, map_location = ("cuda:0","cpu"))
    ##net.load_state_dict(load_weights)


num_epochs = 2
train_model(net, dataloader_dict, criterior, optimizer, num_epochs)
