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
import torch.optim as optim
import time


torch.manual_seed(1234)

torch.set_printoptions(profile="full")

train_dataset = MNIST(root='./data', train=True, download=True, transform=ToTensor())
test_dataset = MNIST(root='./data', train=False, download=True, transform=ToTensor())

def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):

    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

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

train_loader, test_loader = MNIST_loaders()

class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.bn1 = nn.BatchNorm1d(500)
        self.ln1 = nn.LayerNorm(500)
        self.fc2 = nn.Linear(500, 500)
        self.bn2 = nn.BatchNorm1d(500)
        self.ln2 = nn.LayerNorm(500)
        self.fc3 = nn.Linear(500, 10)
        self.num_epochs = 30

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        # x = self.ln1(x)
        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        # x = self.ln2(x)
        x = self.fc3(x)
        return x
    
    def train(self, images, labels):
        for epoch in range(self.num_epochs):
            # model.train()
            # train_loss = 0.0
            # train_correct = 0
            # for images, labels in train_loader:
            images = images.view(-1, 28*28).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            # loss = criterion(outputs, labels)
            loss = torch.log(criterion(outputs.float(), labels))
            # label_oh = nn.functional.one_hot(labels)
            # loss = torch.log(self.loss_fn(outputs.float(), label_oh.float()))

            loss.backward()
            optimizer.step()
            # train_loss += loss.item()
            # _, predicted = torch.max(outputs, 1)
            # train_correct += (predicted == labels).sum().item()

            # train_loss /= len(train_loader)
            # train_accuracy = 100.0 * train_correct / len(train_dataset)

    def predict(self, images):
        # model.eval()
        with torch.no_grad():
        # for images, labels in test_loader:
            images = images.view(-1, 28*28).to(device)
            # labels = labels.to(device)
            outputs = model(images)
        _,fin_out=torch.max(outputs,dim=-1)
        return fin_out
        # return outputs
 


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


model = FCNet()

# criterion = nn.MSELoss()
criterion = multiClassHingeLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

num_epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

x, y = next(iter(train_loader))
x, y = x.cuda(), y.cuda()
st = time.time()
model.train(x, y)
et = time.time()
time1 = (et-st)*1000

print('train error:', 1.0 - model.predict(x).eq(y).float().mean().item())
print('Train time: ', time1)

x_te, y_te = next(iter(test_loader))
x_te, y_te = x_te.cuda(), y_te.cuda()

print('test error:', 1.0 - model.predict(x_te).eq(y_te).float().mean().item())

# from nvitop import Device, GpuProcess, NA, colored

# print(colored(time.strftime('%a %b %d %H:%M:%S %Y'), color='red', attrs=('bold',)))

# devices = Device.cuda.all()  # or `Device.all()` to use NVML ordinal instead
# separator = False
# for device in devices:
#     processes = device.processes()  # type: Dict[int, GpuProcess]

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