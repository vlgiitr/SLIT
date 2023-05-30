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
        return torch.round(10*x.mean(dim = 1))

    def train(self, label, input):
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            input = layer.train(label, input)

# class Cifar10ConvNet(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.conv1 = ConvLayer(3, 16, 3, stride = 1, padding = 0)
#             self.conv2 = ConvLayer(16, 32, 3, stride = 1, padding = 0)
#             self.conv3 = ConvLayer(32, 64, 3, stride = 1, padding = 0)
#             self.conv4 = ConvLayer(64, 128, 3, stride = 1, padding = 0)
#             self.conv5 = ConvLayer(128, 128, 3, stride = 1, padding = 0)
#             self.conv6 = ConvLayer(128, 128, 3, stride = 1, padding = 0)
#             self.conv7 = ConvLayer(128, 128, 3, stride = 1, padding = 0)
#             self.conv8 = ConvLayer(128, 64, 3, stride = 1, padding = 0)
#             self.conv9 = ConvLayer(64, 32, 3, stride = 1, padding = 0)
#             self.conv10 = ConvLayer(32, 16, 3, stride = 1, padding = 0)
#             self.conv11 = ConvLayer(16, 8, 3, stride = 1, padding = 0)
#             self.conv12 = ConvLayer(8, 4, 3, stride = 1, padding = 0)
#             self.layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8, self.conv9, self.conv10, self.conv11, self.conv12]

#         def train(self, label, input):
#             for i, layer in enumerate(self.layers):
#                 print('training layer', i, '...')
#                 input = layer.train(label, input)

#         def forward(self, input):
#             x = self.conv1(input)
#             x = self.conv2(x)
#             x = self.conv3(x)
#             x = self.conv4(x)
#             x = self.conv5(x)
#             x = self.conv6(x)
#             x = self.conv7(x)
#             x = self.conv8(x)
#             x = self.conv9(x)
#             x = self.conv10(x)
#             x = self.conv11(x)
#             x = self.conv12(x)
#             return x


class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.tanh = torch.nn.Tanh()
        self.opt = Adam(self.parameters(), lr=0.1)
        self.threshold = 2.0
        self.num_epochs = 2000
        self.loss_fn = torch.nn.MSELoss()
        self.bn = torch.nn.BatchNorm1d(500)
        self.ln = torch.nn.LayerNorm(500)

    def forward(self, x, train):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        tanh_o = self.tanh(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))
        out = (tanh_o+1)/2
        return out
    def train(self, label, input):

        for i in tqdm(range(self.num_epochs)):
            out = 10*self.forward(input, True).mean(dim = 1)
            assert torch.all((out >= 0) & (out <= 10))
            if i == 0:
                ic(self.weight.sum())
            loss = torch.log(self.loss_fn(out.float(), label.float()))
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
        ic(self.weight.sum())
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

    net = Net([784, 500, 500])
    x, y = next(iter(train_loader))
    x, y = x.cuda(), y.cuda()
    net.train(y, x)

    print('train error:', 1.0 - net.predict(x).eq(y).float().mean().item())

    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.cuda(), y_te.cuda()

    print('test error:', 1.0 - net.predict(x_te).eq(y_te).float().mean().item())