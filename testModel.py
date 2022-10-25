import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms





use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)
net.classifier[6] = nn.Linear(in_features = 4096, out_features = 2)

load_model(net, save_path)

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

print("Result is:", result)