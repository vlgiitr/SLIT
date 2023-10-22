import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam,AdamW,SGD
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from icecream import ic
import numpy as np
import time

torch.set_printoptions(profile="full")

seed=0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True


def CIFAR10_loaders(train_batch_size=1024, test_batch_size=2000):

    transform = Compose([
        ToTensor(),
        Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),
        # Lambda(lambda x: torch.flatten(x))
        ])

    train_loader = DataLoader(
        CIFAR10('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=1, shuffle=True)

    test_loader = DataLoader(
        CIFAR10('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)
    
    train_cuda_list = []
    count = 0
    for (train_data, train_labels) in train_loader:
        train_cuda_list.append((torch.squeeze(train_data).cuda(), torch.squeeze(train_labels).cuda()))
    cuda_train_loader = DataLoader(train_cuda_list, batch_size = train_batch_size, shuffle = False, num_workers = 0)
    return cuda_train_loader, test_loader





class multiClassHingeLoss(nn.Module):
    def __init__(self, p=2, margin=0.2, weight=None, size_average=True):
        super(multiClassHingeLoss, self).__init__()
        self.p=p
        self.margin=margin
        self.weight=weight#weight for each class, size=n_class, variable containing FloatTensor,cuda,reqiures_grad=False
        self.size_average=size_average
    def forward(self, output, y):#output: batchsize*n_class
   
        output_y=output[torch.arange(0,y.size()[0]).long().cuda(),y.data.cuda()].view(-1,1)#view for transpose 
        #margin - output[y] + output[i]    
        loss=output-output_y+self.margin#contains i=y
        #remove i=y items
        loss[torch.arange(0,y.size()[0]).long().cuda(),y.data.cuda()]=0
        #max(0,_)
        loss[loss<0]=0
        #^p
        if(self.p!=1):
            loss=torch.pow(loss,self.p)
        #add weight
        if(self.weight is not None):
            loss=loss*self.weight
        #sum up
        loss=torch.sum(loss)
        # loss=loss+reg
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
        dtype=None,
        learning_rate: float = 0.01,
        num_epochs = 10
    ) -> None:
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.bn=torch.nn.BatchNorm2d(out_channels)
        # self.drop= torch.nn.Dropout(0.1)
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        # self.opt = SGD(self.parameters(), lr=0.1,weight_decay=1e-6, momentum = 0.9)
        self.opt = Adam(self.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs
        self.loss_fn = multiClassHingeLoss()
        
    def forward(self, x, train):
        x_direction = x /(x.norm(dim=(-1,-2), keepdim=True) + 1e-4) 
        rel_o = self.relu((self.bn(self.conv(x_direction))))
        return torch.squeeze(rel_o)
    
    def train(self,train_loader):
        mem = list()
        lab = list()
        for i in tqdm(range(self.num_epochs)):
            # epoch_time = time.time()
            # batchwise_time = 0
            for j, (inputs, labels) in enumerate(train_loader):
                batch_start_time = time.time()
                labels = torch.squeeze(labels.cuda(), dim = 0)
                out = self.forward(torch.squeeze(inputs.cuda(), dim = 0), True)
                out = out.reshape(out.shape[0],-1)
                m, hw = out.shape
                if hw%10 == 0:
                    out = out
                else:
                    out = out[:, 0:-(hw%10)]
                out = out.view(m, 10, -1)
                out = out.mean(dim = -1)
                loss = torch.log(self.loss_fn(out.float(), torch.squeeze(labels.cuda(),dim=0))+1e-4)
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
                torch.cuda.empty_cache()
                if i==self.num_epochs-1:
                    mem.append(self.forward(torch.squeeze(inputs.cuda()), False).detach().cpu())
                    lab.append(labels.detach().cpu())
                    del inputs
                    del labels
                    torch.cuda.empty_cache()
            #     batchwise_time += time.time() - batch_start_time
            # print("Epoch: ", i, "Loss: ", loss.item(), "Time: ", time.time() - epoch_time, "Batchwise Time: ", batchwise_time)          
        buffer_loader=DataLoader(list(zip(mem,lab)),batch_size=1)
        del lab
        del mem              
        torch.cuda.empty_cache()
        return buffer_loader

class CIFARConvNet(nn.Module):
    def __init__(self):
            super().__init__()
            # self.conv1 = ConvLayer(3, 128, 3, stride = 1, padding = 1, learning_rate = 0.1, num_epochs = 100)#0.01
            # self.conv2 = ConvLayer(128, 128, 3, stride = 1, padding = 1, learning_rate = 0.1, num_epochs = 20)
            # self.conv3 = ConvLayer(128, 128, 3, stride = 1, padding = 1, learning_rate = 0.1, num_epochs = 20)
            # self.conv4 = ConvLayer(128, 64, 3, stride = 1, padding = 1, learning_rate = 0.1, num_epochs = 20)
            # self.pool= torch.nn.AvgPool2d((2,2), stride = 2)
            # self.drop= torch.nn.Dropout(0.1)
           
            self.conv1 = ConvLayer(3, 160, 3, stride = 1, padding = 1,learning_rate=0.01,num_epochs=80)#0.01
            self.conv2 = ConvLayer(160,240, 3, stride = 1, padding = 1,learning_rate=0.01,num_epochs=40)
            self.conv3 = ConvLayer(240,240, 3, stride = 1, padding = 1,learning_rate=0.01,num_epochs=20)
            self.pool= torch.nn.MaxPool2d((2,2), stride = 2)
           
           
           
           
         
            self.layers = []
            self.layers.append(self.conv1.cuda())
            self.layers.append(self.pool.cuda())
            self.layers.append(self.conv2.cuda())
            self.layers.append(self.pool.cuda())
            self.layers.append(self.conv3.cuda())
            # self.layers.append(self.conv4.cuda())
           
    def forward(self, x, train = True):
        x = self.conv1(x, train)
        x =  self.pool(x)
        x = self.conv2(x, train)
        x =  self.pool(x)
        x = self.conv3(x, train)
        # x = self.conv4(x, train)
        torch.cuda.empty_cache()
        return x
    
    def predict(self, x):
        for layer in self.layers:
            # if layer==self.pool or layer==self.drop:
            if layer==self.pool:
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
    
    def train(self,mem_loader):
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            # if layer==self.pool or layer==self.drop:
            if layer==self.pool:
                mem=list()
                lab=list()
                for j, data in enumerate(mem_loader):
                   inputs, labels = data
                   inputs=torch.squeeze(inputs,dim=0)
                   labels=torch.squeeze(labels,dim=0)
                   mem.append(layer(inputs).detach())
                   lab.append(labels)
                   del inputs
                   del labels
                   torch.cuda.empty_cache()
                mem_loader=DataLoader(list(zip(mem,lab)),batch_size=1)
                del lab
                del mem
                torch.cuda.empty_cache()
            else:
                    mem_loader = layer.train(mem_loader)
            
 
                    
def get_n_params(model):
    pp, num = 0, 0 
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
        num += 1
    return pp


if __name__ == "__main__":
  
        torch.manual_seed(1234)
        train_loader, test_loader =CIFAR10_loaders()

        net = CIFARConvNet()
        print('number of parameters:', get_n_params(net))
    
        net.train(train_loader)

        train_acc=0
        for data in train_loader:
    
                x,y=data
                x, y = torch.squeeze(x.cuda(),dim=0), torch.squeeze(y.cuda(),dim=0)
                train_acc+=net.predict(x).eq(y).float().sum().item()
    
        print('train error:', 1.0 - (train_acc/50000))
    
        test_acc=0
        for data in test_loader:
            with torch.no_grad():
                x_te,y_te=data
                test_acc+=net.predict(torch.squeeze(x_te.cuda(),dim=0)).eq(torch.squeeze(y_te.cuda(),dim=0)).float().sum().item()
        print('test error:', 1.0 - (test_acc/10000))