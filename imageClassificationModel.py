import numpy as np
import torch
import torchvision
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import json

use_pretrained = True
net = models.vgg16(pretrained = use_pretrained)
net.eval()
print(net)


#Class tien xu ly anh 
class BaseTransform():
    def __init__(self, resize, mean, std):
        self.base_transform = transforms.Compose(
            [transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)]
        )

    def __call__(self, img):
        return self.base_transform(img)

image_file_path = "./1.jpg"
img = Image.open(image_file_path)

plt.imshow(img)
plt.show()

resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

##transform = BaseTransform(resize, mean, std) #tao ra transform
##img_transformed = transform(img) #transform input

#(channels, height, width) -> (height, width, channels)
#clip(0,1) cac gia tri pixel nam giua 0 va 1

##img_transformed = img_transformed.numpy().transpose(1,2,0)
##img_transformed = np.clip(img_transformed, 0, 1)

##plt.imshow(img_transformed)
##plt.show()

class Predictor():
    def __init__(self,class_index):
        self.class_index = class_index

    def predict_max(self,out):
        #detach() dung de tach lop cuoi cung ra khoi network
        #chuyen ve thanh dang numpy
        maxid = np.argmax(out.detach().numpy())
        predicted_label_name =self.class_index[str(maxid)]

        return predicted_label_name

class_index = json.load(open('./imagenet_class_index.json','r'))  
predictor = Predictor(class_index)

img = Image.open(image_file_path)

transform = BaseTransform(resize, mean, std)
img_transformed = transform(img)
img_transformed = img_transformed.unsqueeze_(0)

out = net(img_transformed)
result = predictor.predict_max(out)

print("Result is:", result[1])