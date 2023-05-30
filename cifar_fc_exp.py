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
from torch_lr_finder import LRFinder
import numpy as np
import time

torch.set_printoptions(profile="full")

seed=0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def MNIST_loaders(train_batch_size=2000, test_batch_size=10000):

    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        #Lambda(lambda x: torch.flatten(x))
        ])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader

def CIFAR10_loaders(train_batch_size=5000, test_batch_size=10000):

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

    return train_loader, test_loader

# def overlay_y_on_x(x, y):
#     """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
#     """
#     x_ = x.clone()
#     x_[:, :10] *= 0.0
#     x_[range(x.shape[0]), y] = x.max()
#     return x_

class multiClassHingeLoss(nn.Module):
    def __init__(self, p=2, margin=0.2, weight=None, size_average=True):
        super(multiClassHingeLoss, self).__init__()
        self.p=p
        self.margin=margin
        self.weight=weight#weight for each class, size=n_class, variable containing FloatTensor,cuda,reqiures_grad=False
        self.size_average=size_average
    def forward(self, output, y):#output: batchsize*n_class
        #print(output.requires_grad)
        #print(y.requires_grad)
        # output=nn.functional.softmax(output,-1)
        output_y=output[torch.arange(0,y.size()[0]).long().cuda(),y.data.cuda()].view(-1,1)#view for transpose
       
        #margin - output[y] + output[i]
        
        loss=output-output_y+self.margin#contains i=y

        # ic(loss.size())
        # ic(loss[0])
        # ic(output[0])
        # ic(output_y[0])

        #remove i=y items
        loss[torch.arange(0,y.size()[0]).long().cuda(),y.data.cuda()]=0
        # ic(loss[0])
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
    

# class Net(torch.nn.Module):

#     def __init__(self, dims):
#         super().__init__()
#         self.layers = []
#         self.num_layers = len(dims) - 1
#         for d in range(len(dims) - 1):
#             self.layers += [Layer(dims[d], dims[d + 1]).cuda()]
#         # print(self.layers)
        
#     def predict(self, x):
#         for layer in self.layers:
#             x = layer(x, False)
#         m,hw=x.shape
#         x=x.view(m,-1,hw//10)
#         x= x.mean(dim = -1)
#         _,fin_out=torch.max(x,dim=-1)
#         return fin_out

#     def train(self, label, input):
#         for i, layer in enumerate(self.layers):
#             print('training layer', i, '...')
#             input = layer.train(label, input)



class LinearLayer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.0005) #, lr=0.01
        self.threshold = 2.0
        self.num_epochs = 10

        self.loss_fn = multiClassHingeLoss()# multiClassHingeLoss()
        self.bn = torch.nn.BatchNorm1d(out_features)
        self.ln = torch.nn.LayerNorm(out_features)

    def forward(self, x, train):
        x_direction = x
        # /  (x.norm(2, 1, keepdim=True) + 1e-4)
        x_direction = x_direction.reshape(x_direction.shape[0], -1)
        rel_o = self.relu(self.bn(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0)))
        
        
        return rel_o
    
    def train(self, label, input):

        initial_sum, final_sum = 0, 0
        for k in list(self.parameters()):
            initial_sum += k.sum().item()
        ic(initial_sum)
        for i in tqdm(range(self.num_epochs)):
            
            out = self.forward(input, True)
            m,hw=out.shape
        
            out=out.view(m,10,-1)
            out = out.mean(dim = -1)
            # if(i==1):
            #     ic(label[0])
            #     ic(out[0])
            # if(i==self.num_epochs-1):
            #     ic(out[0])
            # assert torch.all((out >= 0) & (out <= 10))
            # label_oh = nn.functional.one_hot(label)
            # loss = self.loss_fn(out.float(), label)
            loss = torch.log(self.loss_fn(torch.softmax(out.float(),dim=-1), label))
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
           
        for t in list(self.parameters()):
            final_sum += t.sum().item()
        ic(final_sum)
        ic(out[:10])
        ic(label[:10])    
        return self.forward(input, False).detach()
        
            

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
        self.drop= torch.nn.Dropout(0.01)
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        # self.opt = SGD(self.parameters(), lr=0.1, momentum = 0.9)
        self.opt = Adam(self.parameters(), lr=0.003)
        self.threshold = 2.0
        self.num_epochs = 100
        self.loss_fn = multiClassHingeLoss()
        
    def forward(self, x, train):
        x_direction = x 
        # /(x.norm(dim=(-1,-2), keepdim=True) + 1e-4)
        
        rel_o = self.relu((self.bn(self.conv(x_direction))))
        # rel_o=self.drop(rel_o)
        # out = (tanh_o+1)/2
  
        return rel_o
    
    def train(self, label, input):
        initial_sum, final_sum = 0, 0
        for k in list(self.parameters()):
            initial_sum += k.sum().item()
        ic(initial_sum)
        for i in tqdm(range(self.num_epochs)):
            
            self.opt.zero_grad()
            out = self.forward(input, True)
            # out=out.permute(0,2,3,1)
            out=out.reshape(out.shape[0],-1)
       
            m,hw=out.shape
            out=torch.nn.functional.pad(out,pad=(0,(10-(hw%10))),mode='constant',value=0)
            
            # out=out[:,0:-(hw%10)]
           
            out=out.view(m,10,-1)
            out=out.mean(dim=-1)
  
            
            
            
            
            # out=out.resize(m,10,h//10).mean(dim = -1)
            # out= out.mean(dim = -1)
            
            
            loss = torch.log(self.loss_fn(out.float(), label)) 
            loss.backward()
            self.opt.step()
            if i == (self.num_epochs)/2 :
                    for t in list(self.parameters()):
                        final_sum += t.sum().item()
                    ic(final_sum)
            
        for t in list(self.parameters()):
            final_sum += t.sum().item()
        ic(final_sum)
        ic(out[:10])
        ic(label[:10])    
        return self.forward(input, False).detach()

class MNISTConvNet(nn.Module):
    def __init__(self):
            super().__init__()
            self.conv1 = ConvLayer(3, 32, 3, stride = 1, padding = 0)
            self.conv2 = ConvLayer(32, 32, 3, stride = 1, padding = 0)
            # self.maxpool1= torch.nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
            # self.drop= torch.nn.Dropout(0.25)
            self.conv3 = ConvLayer(32, 64, 3, stride = 1, padding = 0)
            self.conv4 = ConvLayer(64, 64, 3, stride = 1, padding = 0)
            self.conv4 = ConvLayer(64, 64, 3, stride = 3, padding = 0)
            # self.conv5 = ConvLayer(64, 10, 3, stride = 1, padding = 0)
            # self.pool2= torch.nn.MaxPool2d(2,2)
            # self.drop2=torch.nn.Dropout(0.25)
            self.lin1 = LinearLayer(4096, 10)
            
            # self.lin2 = LinearLayer(2000, 500)
            # self.lin3 = LinearLayer(500, 10)
            self.layers = []
            self.layers.append(self.conv1.cuda())
            self.layers.append(self.conv2.cuda())
            self.layers.append(self.conv3.cuda())
            self.layers.append(self.conv4.cuda())
            # self.layers.append(self.conv5.cuda())
            self.layers.append(self.lin1.cuda())
            # self.layers.append(self.lin2.cuda())
            # self.layers.append(self.lin3.cuda())
    
    def forward(self, x, train = True):
        x = self.conv1(x, train)
        x = self.conv2(x, train)
        # x = self.maxpool1(x)
        # x=  self.drop(x)
        x = self.conv3(x, train)
        x = self.conv4(x, train)
        # x = self.conv5(x, train)
        # x = self.pool2(x)
        # x=  self.drop(x)
        # x = torch.flatten(x, start_dim = 1)
        x = self.lin1(x, train)
        # x = self.lin2(x, train)
        # x = self.lin3(x, train)
        return x
    
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x, False)
        # x = torch.flatten(x, start_dim = 1)
        # x=x.permute(0,2,3,1)
        # # x=x.reshape(x.shape[0],-1)
        # m,hw=x.shape
        # x=x[:,0:-(hw%10)]
        # x=x.view(m,10,-1)
        # x= x.mean(dim = -1)
        _,fin_out=torch.max(x,dim=-1)
        return fin_out
    
    def train(self, label, input):
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            input = layer.train(label, input.detach())

            

    
def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()
    
    
if __name__ == "__main__":
    torch.manual_seed(1234)
    train_loader, test_loader = CIFAR10_loaders()

    net = MNISTConvNet()
    x, y = next(iter(train_loader))
    x, y = x.cuda(), y.cuda()
    st = time.time()
    net.train(y, x)
    et = time.time()
    time1 = (et-st)*1000
    print('train error:', 1.0 - net.predict(x).eq(y).float().mean().item())
    print('Train time: ', time1)
    
    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.cuda(), y_te.cuda()

    print('test error:', 1.0 - net.predict(x_te).eq(y_te).float().mean().item())
    
#     from nvitop import Device, GpuProcess, NA, colored
# print(colored(time.strftime('%a %b %d %H:%M:%S %Y'), color='red', attrs=('bold',)))

# devices = Device.cuda.all()  # or `Device.all()` to use NVML ordinal instead
# separator = False
# for device in devices:
#     processes = device.processes()  

#     print(colored(str(device), color='green', attrs=('bold',)))
#     print(colored('  - Fan speed:       ', color='blue', attrs=('bold',)) + f'{device.fan_speed()}%')
#     print(colored('  - Temperature:     ', color='blue', attrs=('bold',)) + f'{device.temperature()}C')
#     print(colored('  - GPU utilization: ', color='blue', attrs=('bold',)) + f'{device.gpu_utilization()}%')
#     print(colored('  - Total memory:    ', color='blue', attrs=('bold',)) + f'{device.memory_total_human()}')
#     print(colored('  - Used memory:     ', color='blue', attrs=('bold',)) + f'{device.memory_used_human()}')
#     print(colored('  - Free memory:     ', color='blue', attrs=('bold',)) + f'{device.memory_free_human()}')
#     if len(processes) > 0:
#         processes = GpuProcess.take_snapshots(processes.values(), failsafe=True)
#         processes.sort(key=lambda process: (process.username, process.pid))

#         print(colored(f'  - Processes ({len(processes)}):', color='blue', attrs=('bold',)))
#         fmt = '    {pid:<5}  {username:<8} {cpu:>5}  {host_memory:>8} {time:>8}  {gpu_memory:>8}  {sm:>3}  {command:<}'.format
#         print(colored(fmt(pid='PID', username='USERNAME',
#                           cpu='CPU%', host_memory='HOST-MEM', time='TIME',
#                           gpu_memory='GPU-MEM', sm='SM%',
#                           command='COMMAND'),
#                       attrs=('bold',)))
#         for snapshot in processes:
#             print(fmt(pid=snapshot.pid,
#                       username=snapshot.username[:7] + ('+' if len(snapshot.username) > 8 else snapshot.username[7:8]),
#                       cpu=snapshot.cpu_percent, host_memory=snapshot.host_memory_human,
#                       time=snapshot.running_time_human,
#                       gpu_memory=(snapshot.gpu_memory_human if snapshot.gpu_memory_human is not NA else 'WDDM:N/A'),
#                       sm=snapshot.gpu_sm_utilization,
#                       command=snapshot.command))
#     else:
#         print(colored('  - No Running Processes', attrs=('bold',)))

#     if separator:
#         print('-' * 120)
#     separator = True
