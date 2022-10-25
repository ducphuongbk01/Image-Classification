import torch
import torch.nn as nn
import torch.nn.functional as fnc


####### Build model by nn.Module#########
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() #Goi ham khoi tai init cua class me nn.Module
        self.conv1 = nn.Conv2d(1,6,3) #input_ch, output_ch, stride
        self.conv2 = nn.Conv2d(6,16,3) 
        self.fc1 = nn.Linear(16*6*6,120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = fnc.relu(x)
        x = fnc.max_pool2d(x, (2,2))
        x = self.conv2(x)
        x = fnc.relu(x)
        x = fnc.max_pool2d(x, 2)
        x = x.view(-1, self.num_flat_feature(x))
        x = fnc.relu(self.fc1(x))
        x = fnc.relu(self.fc2(x))
        x = fnc.relu(self.fc3(x))

        return x


    def num_flat_feature(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
input_image = torch.rand(1,1,32,32)
output = net(input_image)
print(output.size())

######### Build Model by nn.Sequential ######## 
class Flattern(nn.Module):
    def forward(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return x.view(-1, num_features)



net = nn.Sequential()
net.add_module("Conv1", nn.Conv2d(1, 6, 3))
net.add_module("reLU1", nn.ReLU())
net.add_module("Maxpooling1", nn.MaxPool2d(2))

net.add_module("Conv2", nn.Conv2d(6, 16, 3))
net.add_module("reLU2", nn.ReLU())
net.add_module("Maxpooling2", nn.MaxPool2d(2))
net.add_module("Flattern", Flattern())

net.add_module("Fc1", nn.Linear(16*6*6,120))
net.add_module("Fc2", nn.Linear(120,84))
net.add_module("Fc3", nn.Linear(84,10))

input_image = torch.rand(1,1,32,32)
output = net(input_image)
print(output.size())

