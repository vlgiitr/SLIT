import torch
import torch.nn.functional as F

from torch import nn
from einops import rearrange

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

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Layer(dim, hidden_dim),
            nn.GELU(),
            nn.Layer(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Layer(dim, dim * 3, bias=False)
        self.to_out = nn.Layer(dim, dim)

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img, mask=None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.transformer(x, mask)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)
    
if __name__ == '__main__':
    torch.manual_seed(1234)
    