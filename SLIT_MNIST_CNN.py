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

seed=0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True


def MNIST_loaders(train_batch_size=20000, test_batch_size=10000):

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
    
    train_cuda_list = []
    for (train_data, train_labels) in train_loader:
        train_cuda_list.append((train_data.cuda(), train_labels.cuda()))
    cuda_train_loader = DataLoader(train_cuda_list, batch_size = 1, shuffle = False, num_workers = 0)
    
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
        dtype=None
    ) -> None:
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.bn=torch.nn.BatchNorm2d(out_channels)
        # self.drop= torch.nn.Dropout(0.1)
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        # self.opt = SGD(self.parameters(), lr=0.1,weight_decay=1e-6, momentum = 0.9)
        self.opt = Adam(self.parameters(), lr=0.01)
        self.threshold = 2.0
        self.num_epochs =30
        self.loss_fn = multiClassHingeLoss()
        
    def forward(self, x, train):
        x_direction = x /(x.norm(dim=(-1,-2), keepdim=True) + 1e-4)
        
        rel_o = self.relu((self.bn(self.conv(x_direction))))
        # rel_o=self.drop(rel_o)
        # out = (tanh_o+1)/2
  
        return rel_o
    
    def train(self,train_loader):
        initial_sum, final_sum = 0, 0
        for k in list(self.parameters()):
            initial_sum += k.sum().item()
        # ic(initial_sum)
        mem = list()
        lab = list()
        ic(len(train_loader.dataset))
        err=0
        for i in tqdm(range(self.num_epochs)):
            batch_time = 0
            epoch_start = time.time()
            for j, data in enumerate(train_loader):
                batch_start = time.time()
        # Every data instance is an input + label pair
                inputs, labels = data
                inputs=torch.squeeze(inputs.cuda(),dim=0)
                labels=torch.squeeze(labels.cuda(),dim=0)
                self.opt.zero_grad()
                out = self.forward(inputs, True)
                out=out.reshape(out.shape[0],-1)
                m,hw=out.shape
                if hw%10==0:
                    out=out
                else:
                    out=out[:,0:-(hw%10)]
          
 
                out=out.view(m,10,-1)
                out=out.mean(dim=-1)
                
                loss = torch.log(self.loss_fn(out.float(), labels))
        
                loss.backward()
                self.opt.step()
                
        
                if i==self.num_epochs-1:
                    
                    mem.append(self.forward(inputs, False).detach())
                    lab.append(labels)
                    _,inter_out=torch.max(out,dim=-1)
                    err+=inter_out.eq(labels).float().sum().item()
                    
            
                   
        buffer_loader=DataLoader(list(zip(mem,lab)),batch_size=1)
        
        del lab
        del mem
        torch.cuda.empty_cache()
        
        
        return buffer_loader

class MNISTConvNet(nn.Module):
    def __init__(self):
            super().__init__()
            self.conv1 = ConvLayer(1, 32, 3, stride = 1, padding = 0)
            self.conv2 = ConvLayer(32,32, 3, stride = 3, padding = 0)
            self.conv3 = ConvLayer(32,64, 3, stride = 1, padding = 0)
         
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
            mem_loader = layer.train(mem_loader)
    
    
if __name__ == "__main__":
    torch.manual_seed(1234)
    train_loader, test_loader =MNIST_loaders()

    net = MNISTConvNet()

    st = time.time()
    net.train(train_loader)
    et = time.time()
    time1 = (et-st)*1000
    output=0
    for data in train_loader:
        x,y=data
        x, y = torch.squeeze(x,dim=0), torch.squeeze(y,dim=0)
        output+=net.predict(x).eq(y).float().sum().item()

    print('train error:', 1.0 - (output/60000))

    output=0
    for data in test_loader:
        x_te,y_te=data
        x_te, y_te = torch.squeeze(x_te.cuda(),dim=0), torch.squeeze(y_te.cuda(),dim=0)
        output+=net.predict(x_te).eq(y_te).float().sum().item()

    print('test error:', 1.0 - (output/10000))
    print('Train time: ', time1)
    

    from nvitop import Device, GpuProcess, NA, colored

    print(colored(time.strftime('%a %b %d %H:%M:%S %Y'), color='red', attrs=('bold',)))

    devices = Device.cuda.all()  # or `Device.all()` to use NVML ordinal instead
    separator = False
    for device in devices:
        processes = device.processes()  
        print(colored(str(device), color='green', attrs=('bold',)))
        print(colored('  - Fan speed:       ', color='blue', attrs=('bold',)) + f'{device.fan_speed()}%')
        print(colored('  - Temperature:     ', color='blue', attrs=('bold',)) + f'{device.temperature()}C')
        print(colored('  - GPU utilization: ', color='blue', attrs=('bold',)) + f'{device.gpu_utilization()}%')
        print(colored('  - Total memory:    ', color='blue', attrs=('bold',)) + f'{device.memory_total_human()}')
        print(colored('  - Used memory:     ', color='blue', attrs=('bold',)) + f'{device.memory_used_human()}')
        print(colored('  - Free memory:     ', color='blue', attrs=('bold',)) + f'{device.memory_free_human()}')
        if len(processes) > 0:
            processes = GpuProcess.take_snapshots(processes.values(), failsafe=True)
            processes.sort(key=lambda process: (process.username, process.pid))

            print(colored(f'  - Processes ({len(processes)}):', color='blue', attrs=('bold',)))
            fmt = '    {pid:<5}  {username:<8} {cpu:>5}  {host_memory:>8} {time:>8}  {gpu_memory:>8}  {sm:>3}  {command:<}'.format
            print(colored(fmt(pid='PID', username='USERNAME',
                            cpu='CPU%', host_memory='HOST-MEM', time='TIME',
                            gpu_memory='GPU-MEM', sm='SM%',
                            command='COMMAND'),
                        attrs=('bold',)))
            for snapshot in processes:
                print(fmt(pid=snapshot.pid,
                        username=snapshot.username[:7] + ('+' if len(snapshot.username) > 8 else snapshot.username[7:8]),
                        cpu=snapshot.cpu_percent, host_memory=snapshot.host_memory_human,
                        time=snapshot.running_time_human,
                        gpu_memory=(snapshot.gpu_memory_human if snapshot.gpu_memory_human is not NA else 'WDDM:N/A'),
                        sm=snapshot.gpu_sm_utilization,
                        command=snapshot.command))
        else:
            print(colored('  - No Running Processes', attrs=('bold',)))

        if separator:
            print('-' * 120)
        separator = True
        
    