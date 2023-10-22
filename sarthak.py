import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam,AdamW,SGD
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from icecream import ic
import numpy as np
import time
from vit import MyViT

def MNIST_loaders(train_batch_size=5000, test_batch_size=10000):

    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))
        ])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size = train_batch_size, shuffle = False, num_workers = 0, pin_memory = True)

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False, num_workers = 0, pin_memory = True)

    train_cuda_list = []
    for (train_data, train_labels) in train_loader:
        train_cuda_list.append((train_data.cuda(), train_labels.cuda()))
    cuda_train_loader = DataLoader(train_cuda_list, batch_size = 1, shuffle = False, num_workers = 0)
    return cuda_train_loader, test_loader

def CIFAR10_loaders(train_batch_size=25000, test_batch_size=10000):

    transform = Compose([
        ToTensor(),
        Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),
        # Lambda(lambda x: torch.flatten(x))
        ])

    train_loader = DataLoader(
        CIFAR10('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        CIFAR10('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)
    
    # train_cuda_list = []
    # for (train_data, train_labels) in train_loader:
    #     train_cuda_list.append((train_data.cuda(), train_labels.cuda()))
    # cuda_train_loader = DataLoader(train_cuda_list, batch_size = 1, shuffle = False, num_workers = 0)
    return train_loader, test_loader

class multiClassHingeLoss(nn.Module):
    def __init__(self, p=2, margin=0.2, weight=None, size_average=True):
        super(multiClassHingeLoss, self).__init__()
        self.p=p
        self.margin=margin
        self.weight=weight#weight for each class, size=n_class, variable containing FloatTensor,cuda,reqiures_grad=False
        self.size_average=size_average
    def forward(self, output, y):#output: batchsize*n_class
        output_y=output[torch.arange(0,y.size()[0]).long().cuda(),y.data.cuda()].view(-1,1)#view for transpose
        loss=output-output_y+self.margin#contains i=y
        loss[torch.arange(0,y.size()[0]).long().cuda(),y.data.cuda()]=0
        loss[loss<0]=0
        if(self.p!=1):
            loss=torch.pow(loss,self.p)
        if(self.weight is not None):
            loss=loss*self.weight
        loss=torch.sum(loss)
        if(self.size_average):
            loss/=output.size()[0]#output.size()[0]
        return loss

class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ) -> None:
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.bn=torch.nn.BatchNorm2d(out_channels)
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.opt = Adam(self.parameters(), lr=0.007)
        self.threshold = 2.0
        self.num_epochs = 2
        self.loss_fn = multiClassHingeLoss()
        
    def forward(self, x):
        x_direction = x /(x.norm(dim=(-1,-2), keepdim=True) + 1e-4)
        return self.relu((self.bn(self.conv(x_direction.cuda()))))
    
    def train(self, train_loader):
        mem = []
        lab = []
        for i in tqdm(range(self.num_epochs)):
            epoch_start = time.time()
            batch_only_time = 0
            for (inputs, labels) in train_loader:
                batch_start = time.time()
                inputs, labels = torch.squeeze(inputs.cuda(), dim=0), torch.squeeze(labels.cuda(), dim=0)        
                out = self.forward(inputs)
                out = out.view(out.shape[0],-1)
                m, hw = out.shape
                if hw % 10 == 0:
                    out = out
                else:
                    out = out[:, 0 : -(hw%10)]
                out = out.view(m, 10, -1)
                out = out.mean(dim = -1)
                loss = torch.log(self.loss_fn(out.float(), labels.cuda()))
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
                inputs.cpu()
                labels.cpu()

                if i==self.num_epochs-1:
                    mem.append(self.forward(inputs).detach())
                    lab.append(labels.detach())
                batch_end = time.time()
                batch_only_time += batch_end - batch_start
            epoch_end = time.time()
            print("Epoch {} completed in {} seconds".format(i, epoch_end - epoch_start))
            print("Batch time: {}".format(batch_only_time))

        buffer_loader = DataLoader(list(zip(mem, lab)), batch_size = 1)
        
        del lab
        del mem
        torch.cuda.empty_cache()
        
        return buffer_loader

class MNISTConvNet(nn.Module):
    def __init__(self):
            super().__init__()
            self.conv1 = ConvLayer(3, 32, 3, stride = 1, padding = 0)
            self.conv2 = ConvLayer(32, 32, 3, stride = 3, padding = 0)
            self.conv3 = ConvLayer(32,64, 3, stride = 1, padding = 0)
            self.pool= torch.nn.MaxPool2d(2,2)
            self.drop= torch.nn.Dropout(0.1)
           
            self.layers = []
            self.layers.append(self.conv1.cuda())
            self.layers.append(self.conv2.cuda())
            self.layers.append(self.conv3.cuda())

    def forward(self, x, train = True):
        x = self.conv1(x, train)
        x = self.conv2(x, train)
        x = self.conv3(x, train)
        return x
    
    def predict(self, x):
        for layer in self.layers:
            if layer==self.pool or layer==self.drop:
                x=layer(x)
            else:
                x = layer.forward(x, False)
        x = torch.flatten(x, start_dim = 1)
        m,hw=x.shape
       
        if hw%10==0:
                x=x
        else:
                x=x[:,0:-(hw%10)]
   
        x=x.view(m,10,-1)
        x= x.mean(dim = -1)
        _,fin_out=torch.max(x,dim=-1)
        return fin_out
    


    def train(self, mem_loader):
        i=0
        for layer in self.layers:
            print('training layer', i, '...')
            mem_loader = layer.train(mem_loader)
            i+=1

if __name__ == "__main__":
    train_loader, test_loader = MNIST_loaders()
    net = MyViT((1, 28, 28)).cuda()
    net.train(train_loader)