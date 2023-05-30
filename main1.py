import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
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

def MNIST_loaders(train_batch_size=5000, test_batch_size=10000):

    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
        ])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True, drop_last = True)

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False, drop_last = True)

    return train_loader, test_loader

def CIFAR10_loaders(train_batch_size=10000, test_batch_size=10000):

    transform = Compose([
        ToTensor(),
        Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),])

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

class Net(torch.nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.layers = []
        self.num_layers = len(dims) - 1
        for d in range(len(dims) - 1):
            self.layers += [LinearLayer(dims[d], dims[d + 1]).cuda()]
        ic(len(self.layers))

    def predict(self, x):
        for layer in self.layers:
            # layer.eval()
            x = layer(x, False)
        return torch.round(10*x.mean(dim = 1))

    def train(self, label, input):
        for i, layer in enumerate(self.layers):
            # layer.train()
            print('training layer', i, '...')
            input = layer.train(label, input)


class LinearLayer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.tanh = torch.nn.Tanh()
        self.opt = Adam(self.parameters(), lr=0.05)
        self.threshold = 2.0
        self.num_epochs = 2000
        self.loss_fn = torch.nn.MSELoss(reduction = 'mean')
        
    def forward(self, x, train):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        x_direction = x_direction.reshape(x_direction.shape[0], -1)
        tanh_o = self.tanh(torch.matmul(x_direction, self.weight.T) + self.bias.unsqueeze(0))
        out = (tanh_o+1)/2
        return out
    
    def train(self, label, input):
        for i in tqdm(range(self.num_epochs)):
            if i == 0:
                ic(self.weight.sum())
            out = 10*self.forward(input, True).mean(dim = 1)
            assert torch.all((out >= 0) & (out <= 10))
            loss = torch.log(self.loss_fn(out.float(), label.float()))
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
        ic(self.weight.sum())
        # exit()
        # ic(out[:10])
        # ic(label[:10]) 
        return self.forward(input, False).detach()

            
class ConvLayer(nn.Conv2d):
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
        super().__init__(in_channels, out_channels, kernel_size, padding = 0, dilation = 1, groups = 1,bias = True, padding_mode = 'zeros')
        self.tanh = torch.nn.Tanh()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.conv.requires_grad_(True)
        # x = self.parameters()
        # ic(x)
        self.opt = SGD(self.parameters(), lr=1, momentum = 0.9)
        self.threshold = 2.0
        self.num_epochs = 150
        self.loss_fn = torch.nn.MSELoss(reduction = 'mean')
        
    def forward(self, x, train):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        tanh_o = self.tanh(self.conv(x_direction))
        return (tanh_o+1)/2
    
    def train(self, label, input):
        for k in self.opt.param_groups:
            for t in k['params']:
                ic(t.shape)
        # exit()
        w = []
        # ic(torch.sum(self.weight))
        for i in tqdm(range(self.num_epochs)):
            self.opt.zero_grad()
            out = 10*self.forward(input, True).mean(dim = (1, 2, 3))
            if i == 0:
                ic(self.weight.sum())
            # ic(i)
            # ic(out.size())
            assert torch.all((out >= 0) & (out <= 10))
            loss = torch.log(self.loss_fn(out.float(), label.float()))
            loss.backward()
            self.opt.step()
            w.append(out.mean(dim=0))
            # w.append(torch.sum(self.weight))
        # ic(torch.sum(self.weight))
        ic(self.weight.sum())
        # exit()
        # ic(out[:10])
        # ic(label[:10]) 
        # ic(w[0])
        # ic(w[100])
        # w = w.cpu()
        # plt.plot(w)
        # plt.show()
        # ic(self.weight.shape)   
        return self.forward(input, False).detach()
    

class MNISTConvNet(nn.Module):
    def __init__(self):
            super().__init__()
            self.conv1 = ConvLayer(1, 32, 3, stride = 1, padding = 0)
            self.conv2 = ConvLayer(32, 32, 3, stride = 1, padding = 0)
            self.conv3 = ConvLayer(32, 64, 3, stride = 1, padding = 0)
            self.conv4 = ConvLayer(64, 64, 3, stride = 5, padding = 0)
            self.linear1 = LinearLayer(1024, 200)
            self.linear2 = LinearLayer(200, 200)
            self.linear3 = LinearLayer(200, 10)
            self.layers = []
            self.layers.append(self.conv1.cuda())
            self.layers.append(self.conv2.cuda())
            self.layers.append(self.conv3.cuda())
            self.layers.append(self.conv4.cuda())
            self.layers.append(self.linear1.cuda())
            self.layers.append(self.linear2.cuda())
            self.layers.append(self.linear3.cuda())
            self.batchnorm = nn.BatchNorm2d(3, affine = False).cuda()
    
    def forward(self, x, train):
        x = self.batchnorm(x)
        x = self.conv1(x, train)
        x = self.conv2(x, train)
        x = self.conv3(x, train)
        x = self.conv4(x, train)
        x = self.linear1(x, train)
        x = self.linear2(x, train)
        x = self.linear3(x, train)
        return x
    
    def predict(self, x):
        for layer in self.layers:
            x = layer(x, False)
        return torch.round(10*x.mean(dim = (1)))
    
    def train(self, label, input):
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            ic(input.shape)
            input = layer.train(label, input)
        
def init_weights_linear(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def init_weights_conv(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

if __name__ == "__main__":
    # torch.manual_seed(1234)
    # train_loader, test_loader = MNIST_loaders()

    # net = Net([784, 500, 500])
    # net.cuda()
    # # net.apply(init_weights_linear)
    # # net.apply(init_weights_conv)
    # print(net)
    # x, y = next(iter(train_loader))
    # x, y = x.cuda(), y.cuda()
    # net.train(y, x)

    # print('train error:', 1.0 - net.predict(x).eq(y).float().mean().item())

    # x_te, y_te = next(iter(test_loader))
    # x_te, y_te = x_te.cuda(), y_te.cuda()

    # print('test error:', 1.0 - net.predict(x_te).eq(y_te).float().mean().item())
    torch.manual_seed(1234)
    train_loader, test_loader = MNIST_loaders()

    net = Net([784, 500, 500])
    x, y = next(iter(train_loader))
    x, y = x.cuda(), y.cuda()
    net.train(y, x)

    print('train error:', 1.0 - net.predict(x).eq(y).float().mean().item())

    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.cuda(), y_te.cuda()

    print('test error:', 1.0 - net.predict(x_te).eq(y_te).float().mean().item())