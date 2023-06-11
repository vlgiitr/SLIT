import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from tqdm import tqdm, trange
from icecream import ic
import torch
import torch.nn as nn
from torch.optim import Adam,AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

np.random.seed(0)
torch.manual_seed(0)

def MNIST_loaders(train_batch_size=5000, test_batch_size=10000):

    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

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

def patchify(images, n_patches):
    n, x = images.shape

    patches = images.view(n, n_patches**2, -1)
    return patches
    # # assert h == w, "Patchify method is implemented for square images only"

    # patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    # patch_size = h // n_patches

    # for idx, image in enumerate(images):
    #     for i in range(n_patches):
    #         for j in range(n_patches):
    #             patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
    #             patches[idx, i * n_patches + j] = patch.flatten()
    # return patches

class multiClassHingeLoss(nn.Module):
    def __init__(self, p=2, margin=0.2, weight=None, size_average=True):
        super(multiClassHingeLoss, self).__init__()
        self.p=p
        self.margin=margin
        self.weight=weight#weight for each class, size=n_class, variable containing FloatTensor,cuda,reqiures_grad=False
        self.size_average=size_average

    def forward(self, output, y):
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

class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        # self.tanh = torch.nn.Tanh()
        self.opt = AdamW(self.parameters(), lr=0.005) #, lr=0.01
        # self.threshold = 2.0
        self.num_epochs = 30
        self.loss_fn = multiClassHingeLoss() #torch.nn.MSELoss() # 
        self.bn = torch.nn.BatchNorm1d(out_features)
        # self.ln = torch.nn.LayerNorm(500)

    def forward(self, x, train):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        # x_direction = x
        out = torch.matmul(x_direction, self.weight.T) + self.bias.unsqueeze(0)
        # out = self.bn(out)
        return torch.relu(out)
    
    def train(self, label, input):
        for i in tqdm(range(self.num_epochs)):
            out = self.forward(input, True)
            # ic(out.size())
            m=out.shape[0]
            out=out.view(m,10,-1)
            out = out.mean(dim = -1)
            loss = torch.log(self.loss_fn(out.float(), label))
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
        # ic(out[:5])
        # ic(label[:5])   
        return self.forward(input, False).detach()

class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads
        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        
        self.num_epochs = 10
        self.opt = AdamW(self.parameters(), lr=0.01) #, lr=0.01
        self.loss_fn = multiClassHingeLoss()

        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences, train):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])
    
    def train(self, label, input):
        for i in tqdm(range(self.num_epochs)):
            out = self.forward(input, True)
            # ic(out.size())
            m,_,_=out.shape
            out=out.view(m,10,-1)
            # out = out.swapaxes(1,2)
            out = out.mean(dim = -1)
            # out = (torch.nn.Tanh(out)+1)/2
            # out = torch.sigmoid(out)
            loss = torch.log(self.loss_fn(out.float(), label))
            loss.backward(retain_graph=True)
            self.opt.step()
            self.opt.zero_grad()
            # ic(out[:5])
        # ic(out[:5])
        # ic(label[:5])   
        return self.forward(input, False).detach() 

class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        # self.mlp = nn.Sequential(
        #     nn.Linear(hidden_d, mlp_ratio * hidden_d),
        #     nn.GELU(),
        #     nn.Linear(mlp_ratio * hidden_d, hidden_d)
        # )
        self.linear1 = Layer(hidden_d, mlp_ratio * hidden_d)
        self.linear2 = Layer(mlp_ratio * hidden_d, hidden_d)
        self.layers = []
        # self.layers.append(self.mhsa.cuda())
        self.layers.append(self.linear1.cuda())
        self.layers.append(self.linear2.cuda())

    def forward(self, x, train):
        out = x + self.mhsa(self.norm1(x),False)
        temp = self.linear1(self.norm2(out),False)
        out = out + self.linear2(temp,False)
        return out

    def train(self, label, input):
        print('training mhsa...')
        input = input + self.mhsa.train(label, self.norm1(input))
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            # ic(input.shape)
            input = layer.train(label, input)
        return self.forward(input, False).detach() 
    
class MyViT(nn.Module):
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=10, n_heads=2, out_d=10):
        # Super constructor
        super(MyViT, self).__init__()
        
        # Attributes
        self.chw = chw # ( C , H , W )
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d
        
        # Input and patches sizes
        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1) Linear mapper
        # self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = Layer(chw[1]*chw[2], n_patches ** 2 * hidden_d)
        
        # 2) Learnable classification token
        # self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
        
        # 3) Positional embedding
        self.register_buffer('positional_embeddings', get_positional_embeddings(n_patches ** 2, hidden_d), persistent=False)
        
        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])
        
        # 5) Classification MLPk
        self.mlp = Layer(self.hidden_d, out_d)
        # self.mlp = nn.Sequential(
        #     nn.Linear(self.hidden_d, out_d),
        #     nn.Softmax(dim=-1)
        # )

    def predict(self, images):
            # Running linear layer tokenization
            # Map the vector corresponding to each patch to the hidden size dimension
            images = self.linear_mapper(images, False)
            
            # Dividing images into patches
            n, x = images.shape
            patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)
            
            
            # Adding classification token to the tokens
            # tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)
            
            # Adding positional embedding
            out = patches + self.positional_embeddings.repeat(n, 1, 1)
            
            # Transformer Blocks
            for block in self.blocks:
                out = block(out, False)
                
            # Getting the classification token only
            # out = out[:, 0]

            out = self.mlp(out, False)
            # ic(out.size())
            # ic(out[1])
            # m,hw=out.shape
            # out=out.view(m,-1,hw//10)
            m=out.shape[0]
            out=out.view(m,10,-1)
            out = out.mean(dim = -1)
            # out = torch.swapaxes(out,1,2)
            # ic(out.size())
            # ic(out[1])
            # out= out.mean(dim = -1)
            _,fin_out=torch.max(out,dim=-1)
            # ic(fin_out[0])
            # ic(out[:5])
            # ic(fin_out[:5])
            return fin_out
            # return self.mlp(out, False) # Map to output dimension, output category distribution

    def train(self, label, input):
        print('training linear mapper...')
        input = self.linear_mapper.train(label, input)

        n, x = input.shape
        input = patchify(input, self.n_patches).to(self.positional_embeddings.device)

        input = input + self.positional_embeddings.repeat(n, 1, 1)

        for i, block in enumerate(self.blocks):
            print('training block', i, '...')
            input = block.train(label,input)
        
        print('training mlp...')
        self.mlp.train(label, input)
    
def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = MyViT((1, 28, 28))
    # Loading data
    # transform = ToTensor()

    # train_set = MNIST(root='./../datasets', train=True, download=True, transform=transform)
    # test_set = MNIST(root='./../datasets', train=False, download=True, transform=transform)

    # train_loader = DataLoader(
    #     MNIST('./data/', train=True,
    #           download=True,
    #           transform=transform),
    #     batch_size=128, shuffle=True)
    

    # test_loader = DataLoader(
    #     MNIST('./data/', train=False,
    #           download=True,
    #           transform=transform),
    #     batch_size=128, shuffle=False)
    
    train_loader, test_loader = MNIST_loaders()

    # Defining model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    model = MyViT((1, 28, 28)).to(device)
    
    # N_EPOCHS = 1
    # LR = 0.005

    # # Training loop
    # optimizer = Adam(model.parameters(), lr=LR)
    # criterion = CrossEntropyLoss()

    x = get_n_params(model)
    ic(x)    

    x, y = next(iter(train_loader))
    x, y = x.cuda(), y.cuda()
    model.train(y,x)
    # for epoch in trange(N_EPOCHS, desc="Training"):
    #     train_loss = 0.0
    #     for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
    #         x, y = batch
    #         x, y = x.to(device), y.to(device)
    #         y_hat = model(x)
    #         loss = criterion(y_hat, y)

    #         train_loss += loss.detach().cpu().item() / len(train_loader)

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #     print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

    # # Test loop
    # with torch.no_grad():
    #     correct, total = 0, 0
    #     test_loss = 0.0
    #     for batch in tqdm(test_loader, desc="Testing"):
    #         x, y = batch
    #         x, y = x.to(device), y.to(device)
    #         y_hat = model(x)
    #         loss = criterion(y_hat, y)
    #         test_loss += loss.detach().cpu().item() / len(test_loader)

    #         correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
    #         total += len(x)
    #     print(f"Test loss: {test_loss:.2f}")
    #     print(f"Test accuracy: {correct / total * 100:.2f}%")

    print('train error:', 1.0 - model.predict(x).eq(y).float().mean().item())
    # print('Train time: ', time1)

    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.cuda(), y_te.cuda()

    print('test error:', 1.0 - model.predict(x_te).eq(y_te).float().mean().item())


if __name__ == '__main__':
    main()