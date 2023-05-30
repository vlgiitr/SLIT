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

torch.set_printoptions(profile="full")

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
        print(self.layers)
        
    def predict(self, x):
        for layer in self.layers:
            x = layer(x, False)
        m,hw=x.shape
        x=x.view(m,-1,hw//10)
        x= x.mean(dim = -1)
        _,fin_out=torch.max(x,dim=-1)
        return fin_out

    def train(self, label, input):
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            input = layer.train(label, input)



class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.tanh = torch.nn.Tanh()
        self.opt = Adam(self.parameters(), lr=0.05)
        self.threshold = 2.0
        self.num_epochs = 1000
        self.loss_fn = multiClassHingeLoss()
        self.bn = torch.nn.BatchNorm1d(500)
        self.ln = torch.nn.LayerNorm(500)

    def forward(self, x, train):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        tanh_o = self.tanh(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))
        out = (tanh_o+1)/2
        return out
    def train(self, label, input):

        initial_sum, final_sum = 0, 0
        for k in list(self.parameters()):
            initial_sum += k.sum().item()
        ic(initial_sum)
        for i in tqdm(range(self.num_epochs)):
            out = self.forward(input, True)
            m,hw=out.shape
            out=out.view(m,-1,hw//10)
            out = out.mean(dim = -1)
            assert torch.all((out >= 0) & (out <= 10))
            loss = self.loss_fn(out.float(), label)
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
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
        
            
    
def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()
    
    
if __name__ == "__main__":
    torch.manual_seed(1234)
    train_loader, test_loader = MNIST_loaders()

    net = Net([784, 500, 500, 400, 400, 400])
    x, y = next(iter(train_loader))
    x, y = x.cuda(), y.cuda()
    net.train(y, x)

    print('train error:', 1.0 - net.predict(x).eq(y).float().mean().item())

    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.cuda(), y_te.cuda()

    print('test error:', 1.0 - net.predict(x_te).eq(y_te).float().mean().item())