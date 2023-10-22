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



def CIFAR10_loaders(train_batch_size=1024, test_batch_size=5000):

    transform = Compose([
        ToTensor(),
        Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),
        # Lambda(lambda x: torch.flatten(x))]
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
    
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.relu = torch.nn.ReLU()
        self.pool=torch.nn.MaxPool2d(2,2)
        self.conv1 = torch.nn.Conv2d(3, 64, 3, stride = 1, padding = 1)
        self.bn1=torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, stride = 1, padding = 1)
        self.bn2=torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, stride = 1, padding = 1)
        self.bn3=torch.nn.BatchNorm2d(128)
        # self.conv4 = torch.nn.Conv2d(64, 64, 3, stride = 5, padding = 0)
        # self.bn4=torch.nn.BatchNorm2d(64)
        self.fc = nn.Linear(32768, 10)
        self.bn = torch.nn.BatchNorm1d(10)
        self.num_epochs = 20

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        # x = self.conv4(x)
        # x = self.bn4(x)
        # x = self.relu(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    def train(self, train_loader):
        for epoch in range(self.num_epochs):
           for j, (images, labels) in enumerate(train_loader): 
            
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
           
            loss = torch.log(criterion(outputs.float(), labels))
            

            loss.backward()
            optimizer.step()
            
    def predict(self, images):
        # model.eval()
        with torch.no_grad():
        
            images = images.to(device)
            
            outputs = model(images)
        _,fin_out=torch.max(outputs,dim=-1)
        return fin_out
    
model = ConvNet()
train_loader, test_loader = CIFAR10_loaders()
# criterion = nn.MSELoss()
criterion = multiClassHingeLoss()
optimizer = Adam(model.parameters(), lr=0.0008)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# x, y = next(iter(train_loader))
# x, y = x.cuda(), y.cuda()
st = time.time()
model.train(train_loader)
et = time.time()
time1 = (et-st)*1000
output=0
for data in train_loader:
        x,y=data
        x, y = torch.squeeze(x.cuda(),dim=0), torch.squeeze(y.cuda(),dim=0)
        output+=model.predict(x).eq(y).float().sum().item()


print('train error:', 1.0 - (output/50000))
print('Train time: ', time1)

# x_te, y_te = next(iter(test_loader))
# x_te, y_te = x_te.cuda(), y_te.cuda()
output=0
for data in test_loader:
        x,y=data
        x, y = torch.squeeze(x.cuda(),dim=0), torch.squeeze(y.cuda(),dim=0)
        output+=model.predict(x).eq(y).float().sum().item()
print('test error:', 1.0 - (output/10000))



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