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
from torchvision import datasets

torch.set_printoptions(profile="full")

train_batch_size=30000

def MNIST_loaders(train_batch_size=30000, test_batch_size=10000):

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

def CIFAR10_loaders(train_batch_size=50000, test_batch_size=10000):

    transform = Compose([
        ToTensor(),
        Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),
        Lambda(lambda x: torch.flatten(x))])

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

def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_

class multiClassHingeLoss(nn.Module):
    def __init__(self, p=2, margin=0.5, weight=None, size_average=True):
        super(multiClassHingeLoss, self).__init__()
        self.p=p
        self.margin=margin
        self.weight=weight#weight for each class, size=n_class, variable containing FloatTensor,cuda,reqiures_grad=False
        self.size_average=size_average
    def forward(self, output, y):#output: batchsize*n_class
        #print(output.requires_grad)
        #print(y.requires_grad)
        
        output_y=output[torch.arange(0,y.size()[0]).long().cuda(),y.data.cuda()].view(-1,1)#view for transpose
        
        # ic(output_y.size())
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
        if(self.size_average):
            loss/=output.size()[0]#output.size()[0]
        return loss
    

class Net(torch.nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.layers = []
        self.num_layers = len(dims) - 1
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1]).cuda()]
        # print(self.layers)
        
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        m,hw=x.shape
        x=x.view(m,-1,hw//10)
        x= x.mean(dim = -1)
        _,fin_out=torch.max(x,dim=-1)
        return fin_out

    def train(self, input):
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            input = layer.train(input)



class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.tanh = torch.nn.Tanh()
        self.opt = Adam(self.parameters(), lr=0.005) #, lr=0.01
        self.threshold = 2.0
        self.num_epochs = 2
        self.loss_fn = multiClassHingeLoss() # torch.nn.MSELoss() # 
        self.bn = torch.nn.BatchNorm1d(out_features)
        self.ln = torch.nn.LayerNorm(out_features)

    def forward(self, x):
        bn = torch.nn.BatchNorm1d(500).cuda()
        # x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        x_direction = x
        # tanh_o = self.tanh(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))
        # out = (tanh_o+1)/2
        out = torch.matmul(x_direction, self.weight.T) + self.bias.unsqueeze(0)
        out = bn(out)
        out = self.ln(out)
        return torch.relu(out)
    
    def train(self, train_loader):
        mem = torch.Tensor([])
        lab = torch.Tensor([])
        for i in tqdm(range(self.num_epochs)):
            epoch_start = time.time()
            batch_only_time = 0
            for (inputs, labels) in train_loader:
                batch_start = time.time()
                inputs, labels = torch.squeeze(inputs.cuda(), dim=0), torch.squeeze(labels.cuda(), dim=0)
                # ic(inputs.size())        
                out = self.forward(inputs)
                out = out.view(out.shape[0],-1)
                m, hw = out.shape
                if hw % 10 == 0:
                    out = out
                else:
                    out = out[:, 0 : -(hw%10)]
                out = out.view(m, 10, -1)
                out = out.mean(dim = -1)
                loss = torch.log(self.loss_fn(out.float(), labels.int().cuda()))
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
                inputs.cpu()
                labels.cpu()

                if i==self.num_epochs-1:
                    # mem.append(self.forward(inputs).detach())
                    # lab.append(labels.detach())
                    mem = torch.cat((mem,self.forward(inputs).detach().cpu()),0)
                    lab = torch.cat((lab,labels.cpu()),0)
                batch_end = time.time()
                batch_only_time += batch_end - batch_start
            epoch_end = time.time()
            print("Epoch {} completed in {} seconds".format(i, epoch_end - epoch_start))
            print("Batch time: {}".format(batch_only_time))

        buffer_loader = DataLoader(list(zip(mem, lab)), batch_size = train_batch_size)
        
        del lab
        del mem
        torch.cuda.empty_cache()
        
        return buffer_loader
            
def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()
    
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
    train_loader, test_loader = MNIST_loaders()
    net = Net([784, 500, 500])
    print(net)
    print(get_n_params(net))
    tryuio   
    st = time.time()
    net.train(train_loader)
    et = time.time()
    time1 = (et-st)*1000

    train_accuracy = 0
    for data in train_loader:
        x,y = data
        x,y = torch.squeeze(x.cuda(), dim=0), torch.squeeze(y.cuda(), dim=0)
        ic(x.size())
        ic(y.size())
        train_accuracy = train_accuracy + net.predict(x).eq(y).float().sum().item()

    print('train error:', 1.0 - train_accuracy/60000)
    print('Train time: ', time1)

    test_accuracy = 0
    for data in test_loader:
        x,y = data
        x,y = torch.squeeze(x.cuda(), dim=0), torch.squeeze(y.cuda(), dim=0)
        # ic(x.size())
        test_accuracy = test_accuracy + net.predict(x).eq(y).float().sum().item()

    print('test error:', 1.0 - test_accuracy/len(test_loader.dataset))


    # from nvitop import Device, GpuProcess, NA, colored

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
    #                         cpu='CPU%', host_memory='HOST-MEM', time='TIME',
    #                         gpu_memory='GPU-MEM', sm='SM%',
    #                         command='COMMAND'),
    #                     attrs=('bold',)))
    #         for snapshot in processes:
    #             print(fmt(pid=snapshot.pid,
    #                     username=snapshot.username[:7] + ('+' if len(snapshot.username) > 8 else snapshot.username[7:8]),
    #                     cpu=snapshot.cpu_percent, host_memory=snapshot.host_memory_human,
    #                     time=snapshot.running_time_human,
    #                     gpu_memory=(snapshot.gpu_memory_human if snapshot.gpu_memory_human is not NA else 'WDDM:N/A'),
    #                     sm=snapshot.gpu_sm_utilization,
    #                     command=snapshot.command))
    #     else:
    #         print(colored('  - No Running Processes', attrs=('bold',)))

    #     if separator:
    #         print('-' * 120)
    #     separator = True
        
    