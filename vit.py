import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from tqdm import tqdm, trange
from icecream import ic
import torch
import torch.nn as nn
from torch.optim import Adam,AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST
from torch.autograd import Variable
import time
from sklearn.model_selection import ParameterGrid, GridSearchCV


np.random.seed(0)
torch.manual_seed(0)

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

def patchify(images, n_patches):
    n, x = images.shape

    patches = images.view(n, n_patches**2, -1)
    return patches

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

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
        self.opt = AdamW(self.parameters(), lr=linear_lr) #, lr=0.01
        self.num_epochs = linear_epochs
        self.loss_fn = multiClassHingeLoss() 
        self.bn = torch.nn.BatchNorm1d(out_features)
        # self.ln = torch.nn.LayerNorm(500)

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        out = torch.matmul(x_direction, self.weight.T) + self.bias.unsqueeze(0)
        return torch.relu(out)
    
    def train(self, train_loader):
        mem = []
        lab = []
        
        for i in (range(self.num_epochs)):
            epoch_start = time.time()
            batch_only_time = 0
            for (inputs, labels) in train_loader:
                batch_start = time.time()
                inputs, labels = torch.squeeze(inputs.cuda(), dim=0), torch.squeeze(labels.cuda(), dim=0)        
                out = self.forward(inputs)
                out = out.view(out.shape[0],-1)
                m, hw = out.shape
                if hw % 10 == 0:
                    out = out
                else:
                    out = out[:, 0 : -(hw%10)]
                out = out.view(m, 10, -1)
                out = out.mean(dim = -1)
                loss = torch.log(self.loss_fn(out.float(), labels.cuda()))
                loss.backward(retain_graph=False)
                self.opt.step()
                self.opt.zero_grad()
                inputs.cpu()
                labels.cpu()

                if i==self.num_epochs-1:
                    fwd = self.forward(inputs).detach()
                    mem.append(fwd)
                    lab.append(labels.detach())
                batch_end = time.time()
                batch_only_time += batch_end - batch_start
            epoch_end = time.time()
            # print(f"linear loss: {loss}")
            # print("Epoch {} completed in {} seconds".format(i, epoch_end - epoch_start))
            # print("Batch time: {}".format(batch_only_time)) 
        buffer_loader = DataLoader(list(zip(mem, lab)), batch_size = 1)
        
        del lab
        del mem
        torch.cuda.empty_cache()
        
        return buffer_loader

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
        
        self.num_epochs = mhsa_epochs
        self.opt = AdamW(self.parameters(), lr=mhsa_lr) #, lr=0.01
        self.loss_fn = multiClassHingeLoss()

        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                seq_result.append(self.softmax(self.q_mappings[head](seq) @ self.k_mappings[head](seq).T / (self.d_head ** 0.5)) @ self.v_mappings[head](seq))
            result.append(torch.hstack(seq_result))
        return torch.stack(result, dim = 0)

    def train(self, train_loader):
        mem = []
        lab = []
        for i in (range(self.num_epochs)):
            epoch_start = time.time()
            batch_only_time = 0
            for (inputs, labels) in train_loader:
                batch_start = time.time()
                inputs, labels = torch.squeeze(inputs.cuda(), dim=0), torch.squeeze(labels.cuda(), dim=0)        
                out = self.forward(inputs)
                out = out.view(out.shape[0],-1)
                m, hw = out.shape
                if hw % 10 == 0:
                    out = out
                else:
                    out = out[:, 0 : -(hw%10)]
                out = out.view(m, 10, -1)
                out = out.mean(dim = -1)
                loss = torch.log(self.loss_fn(out.float(), labels.cuda()))
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
                if i==self.num_epochs-1:
                    mem.append(self.forward(inputs).detach())
                    lab.append(labels.detach())
                batch_end = time.time()
                batch_only_time += batch_end - batch_start
            epoch_end = time.time()
            # ic(loss)
            # print("Epoch {} completed in {} seconds".format(i, epoch_end - epoch_start))
            # print("Batch time: {}".format(batch_only_time))

        buffer_loader = DataLoader(list(zip(mem, lab)), batch_size = 1)
        
        del lab
        del mem
        torch.cuda.empty_cache()
        
        return buffer_loader

class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=2):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.linear1 = Layer(hidden_d, mlp_ratio * hidden_d)
        self.linear1_5 = Layer(mlp_ratio * hidden_d, mlp_ratio * hidden_d)
        self.linear2 = Layer(mlp_ratio * hidden_d, hidden_d)
        self.layers = []
        self.layers.append(self.linear1.cuda())
        self.layers.append(self.linear1_5.cuda())
        self.layers.append(self.linear2.cuda())

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        temp = self.linear1(out)
        temp = self.linear1_5(temp)
        out = out + self.linear2(temp)
        return out
    
    def train(self, train_loader):
        train_loader = self.mhsa.train(train_loader)
        for i, layer in enumerate(self.layers):
            # print('training layer', i, '...')
            train_loader = layer.train(train_loader)
        return train_loader 
    
class MyViT(nn.Module):
    def __init__(self, chw, n_patches=7, n_blocks=1, hidden_d=40, n_heads=2, out_d=10):
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


    def predict(self, images):
            # Running linear layer tokenization
            images = self.linear_mapper(images)
            
            # Dividing images into patches
            n, x = images.shape
            patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)
            
            
            # Adding classification token to the tokens
            # tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)
            
            # Adding positional embedding
            out = patches + self.positional_embeddings.repeat(n, 1, 1)
            
            # Transformer Blocks
            for block in self.blocks:
                out = block(out)

            out = self.mlp(out)
            m=out.shape[0]
            out=out.view(m,10,-1)
            out = out.mean(dim = -1)
            _,fin_out=torch.max(out,dim=-1)
            return fin_out

    def train(self, train_loader):
        # print('training linear mapper...')
        train_loader = self.linear_mapper.train(train_loader)
        
        new_input = []
        new_label = []
        for data in train_loader:
            input,label = data
            input=torch.squeeze(input.cuda(),dim=0)
            n, x = input.shape
            input = patchify(input, self.n_patches).to(self.positional_embeddings.device)
            # input = input + self.positional_embeddings.repeat(n, 1, 1)
            new_input.append(input)
            new_label.append(label)

        train_loader = DataLoader(list(zip(new_input, new_label)), batch_size = 1)
        for i, block in enumerate(self.blocks):
            # print('training block', i, '...')
            input = block.train(train_loader)
        
        # print('training mlp...')
        self.mlp.train(input)        
    
def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result

def get_n_params(model):
    pp, num = 0, 0 
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
        num += 1
    return pp

# linear_epochs = 80
# mhsa_epochs = 7
linear_lr = 0.005
# mhsa_lr = 0.1


if __name__ == '__main__':

    linear_epochs = np.arange(30,81,5)
    mhsa_epochs = np.arange(2,15,2)
    # linear_lr = [0.005]
    mhsa_lr = [0.005]

    # while linear_lr[-1]<0.1:
    #     linear_lr.append(linear_lr[-1]*2)
    while mhsa_lr[-1]<0.1:
        mhsa_lr.append(mhsa_lr[-1]*2)


    param_grid = {'linear_epochs': linear_epochs,
                'mhsa_epochs': mhsa_epochs,
                # 'linear_lr': linear_lr,
                'mhsa_lr': mhsa_lr}

    grid = ParameterGrid(param_grid)

    max_train_acc = 0
    max_test_acc = 0
    best1=0
    best2=0
    best3=0
    best4=0
    i=1
    for params in grid:
        print("Train session: ",i )
        i=i+1
        linear_epochs = params['linear_epochs']
        mhsa_epochs = params['mhsa_epochs']
        # linear_lr = params['linear_lr']
        mhsa_lr = params['mhsa_lr']
        # ic(linear_epochs)
        # ic(mhsa_epochs)
        # ic(linear_lr)
        # ic(mhsa_lr)

        train_loader, test_loader = MNIST_loaders()

        # Defining model and training options
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
        model = MyViT((1, 28, 28)).to(device)


        # x = get_n_params(model)
        # ic(x)
        # st = time.time()
        model.train(train_loader)
        # et = time.time()
        # time1 = (et-st)*1000

        
        train_accuracy = 0
        for data in train_loader:
            with torch.no_grad():
                x,y = data
                x,y = torch.squeeze(x.cuda(), dim=0), torch.squeeze(y.cuda(), dim=0)
                train_accuracy = train_accuracy + model.predict(x).eq(y).float().sum().item()

        # print('train error:', 1.0 - train_accuracy/50000)
        # print('Train time: ', time1)

        test_accuracy = 0
        for data in test_loader:
            with torch.no_grad():
                x,y = data
                x,y = torch.squeeze(x.cuda(), dim=0), torch.squeeze(y.cuda(), dim=0)
                test_accuracy = test_accuracy + model.predict(x).eq(y).float().sum().item()

        if(test_accuracy>max_test_acc):
            max_train_acc = max(max_train_acc,train_accuracy)
            max_test_acc = max(max_test_acc, test_accuracy)
            best1=linear_epochs
            best2=mhsa_epochs
            # best3=linear_lr
            best4=mhsa_lr

        if(i%10==0):
            ic(max_train_acc)
            ic(max_test_acc)
            best1=linear_epochs
            best2=mhsa_epochs
            # best3=linear_lr
            best4=mhsa_lr

        # print('test error:', 1.0 - test_accuracy/len(test_loader.dataset))
    

    train_loader, test_loader = MNIST_loaders()

    print('train error:', 1.0 - max_train_acc/60000)    
    print('test error:', 1.0 - max_test_acc/len(test_loader.dataset))
    ic(best1)
    ic(best2)
    # ic(best3)
    ic(best4)