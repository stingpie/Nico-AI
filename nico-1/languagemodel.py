import torch as T
import torch.nn.functional as F

from torch.nn.attention import SDPBackend, sdpa_kernel

class LanguageModel(T.nn.Module):

    def __init__(self, data_width, emb_size, vocab_size=12800, temporal_division=8,  device='cpu'):
        super().__init__()

        self.device=device

        self.embedding = T.nn.Embedding(vocab_size, data_width, device=self.device)
        self.tvat = TVAT(data_width, emb_size = emb_size, temporal_division = temporal_division, device=self.device)
        self.exbedding = T.nn.Linear(data_width, vocab_size, device=self.device)
        self.kl=0


        opt = T.optim.Adam(list([*self.embedding.parameters(), *self.exbedding.parameters()]))
        for i in range(0):
            F.cross_entropy(self.exbedding(self.embedding(T.arange(vocab_size, device=self.device))), T.diag(T.ones(vocab_size,device=self.device))).backward()
            opt.step()


    def forward(self, x):
        with sdpa_kernel(SDPBackend.MATH):
            #print(":D")
            decoded, loss = self.tvat(self.embedding(x))
            self.kl = self.tvat.kl
            mask = T.where(T.sum(T.abs(decoded), dim=-1,keepdim=True)==0,0,1)


            # technique to lessen repetitive outputs
            #hist_mask = T.zeros((x.size(1),x.size(1)),device=self.device)
            #for i in range(10):
            #    hist_mask = T.diagonal_scatter(hist_mask, T.ones(x.size(1)-i-1,device=self.device)/(2**i), offset=-1-i)
            #decoded -= T.sum(decoded.unsqueeze(-2).expand(-1,-1,x.size(1),-1) * (hist_mask.flip((1,)).reshape(1,x.size(1),x.size(1),1)), dim=-2)*0.5


            return mask*self.exbedding(decoded), loss

    def embed(self, x):
        with sdpa_kernel(SDPBackend.MATH):
            result, len_loss, sections =  self.tvat.encode(self.embedding(x))
            self.kl = self.tvat.kl
            return result, len_loss, sections

    def exbed(self, x, len, correct=None):
        with sdpa_kernel(SDPBackend.MATH):
            result = self.tvat.decode(x, len, z=correct)
            return self.exbedding(result)

'''

from torch.nn.attention import SDPBackend, sdpa_kernel
import torch as T
import torch.nn.functional as F
from tvat import TVAT

class LanguageModel(T.nn.Module):

    def __init__(self, data_width, emb_size, vocab_size=12800, temporal_division=8,  device='cpu'):
        super().__init__()

        self.device=device

        self.embedding = T.nn.Embedding(vocab_size, data_width, device=self.device)
        self.tvat = TVAT(data_width, emb_size = emb_size, temporal_division = temporal_division, device=self.device)
        self.exbedding = T.nn.Linear(data_width, vocab_size, device=self.device)
        self.kl=0


        opt = T.optim.Adam(list([*self.embedding.parameters(), *self.exbedding.parameters()]))
        for i in range(0):
            F.cross_entropy(self.exbedding(self.embedding(T.arange(vocab_size, device=self.device))), T.diag(T.ones(vocab_size,device=self.device))).backward()
            opt.step()


    def forward(self, x):
        with sdpa_kernel(SDPBackend.MATH):
            #print(":D")
            decoded, loss = self.tvat(self.embedding(x))
            self.kl = self.tvat.kl
            mask = T.where(T.sum(T.abs(decoded), dim=-1,keepdim=True)==0,0,1)


            # technique to lessen repetitive outputs
            #hist_mask = T.zeros((x.size(1),x.size(1)),device=self.device)
            #for i in range(10):
            #    hist_mask = T.diagonal_scatter(hist_mask, T.ones(x.size(1)-i-1,device=self.device)/(2**i), offset=-1-i)
            #decoded -= T.sum(decoded.unsqueeze(-2).expand(-1,-1,x.size(1),-1) * (hist_mask.flip((1,)).reshape(1,x.size(1),x.size(1),1)), dim=-2)*0.5


            return mask*self.exbedding(decoded), loss

    def embed(self, x):
        with sdpa_kernel(SDPBackend.MATH):
            result, len_loss, sections =  self.tvat.encode(self.embedding(x))
            self.kl = self.tvat.kl
            return result, len_loss, sections

    def exbed(self, z, x):
        with sdpa_kernel(SDPBackend.MATH):
            result = self.tvat.decode(z, x)
            return self.exbedding(result)
'''

