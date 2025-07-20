# modified from https://avandekleut.github.io/vae/

import torch as T
import torch.nn.functional as F
import math

class PositionalEncoding(T.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, device='cpu'):
        super().__init__()
        self.device=device
        self.dropout = T.nn.Dropout(p=dropout)

        position = T.arange(max_len, device=self.device).unsqueeze(1)
        div_term = T.exp(T.arange(0, d_model, 2, device=self.device) * (-math.log(10000.0) / d_model))
        pe = T.zeros((max_len, 1, d_model), device=self.device)
        pe[:, 0, 0::2] = T.sin(position * div_term)
        pe[:, 0, 1::2] = T.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x= x.transpose(0,1)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x).transpose(0,1)


def drop_tokens(x, probability):
    return x * T.where(T.rand(x.size()[:-1],device=x.device)>probability,0,1).unsqueeze(-1)

def direction(x):
    return T.round(T.clamp(x, min=-1, max=1))

def deNaNed(x):
    return T.where(T.isnan(x),0,x)

def max_norm(x, max):
  return T.where(T.sum(x)>max, F.normalize(x, p=1, dim=-1)*max, x)

class BernoulliMix(T.nn.Module):
  def __init__(self, mix_ratio=0.5, device='cpu'):
    super().__init__()
    self.device=device
    self.B = T.distributions.Bernoulli(probs=mix_ratio)
    if(isinstance(device, str) and "cuda" in device):
      self.B.probs = self.B.probs.cuda()

  def set_probs(self, probs):
    self.B = T.distributions.Bernoulli(probs=probs)
    if(isinstance(self.device, str) and "cuda" in self.device):
      self.B.probs = self.B.probs.cuda()

  def forward(self, a, b):

    if(self.training):
      return T.where(self.B.sample(a.size()).bool(), a, b)
    else:
      return a

def softabs(x,k):
  return (x**2)/(k+T.abs(x))



class TVET(T.nn.Module): # transformer variational encoder across time dimension.
    def __init__(self, data_width, emb_size=None,  temporal_division=8, layer_num=3, nheads = 6, device='cpu'):
        super().__init__()
        self.device=device
        self.emb_size = emb_size if emb_size!=None else data_width
        self.layers = [T.nn.TransformerEncoderLayer(data_width, nheads, dim_feedforward=data_width, batch_first=True, device=self.device )]*(layer_num-1)
        self.expansion = T.nn.Linear(data_width, self.emb_size*2, device=self.device)
        self.var_layer = T.nn.TransformerEncoderLayer(self.emb_size*2, nheads, dim_feedforward=data_width*2, batch_first=True, device=self.device)
        self.poscoder = PositionalEncoding(data_width, max_len=10_000, device=self.device)

        self.expansion_offset=0

        self.temporal_division = temporal_division


        self.N = T.distributions.Normal(0, 1)
        if(isinstance(device, str) and "cuda" in device):
            self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = self.poscoder(x)

        for layer in self.layers:
            x = layer(x, src_mask = T.nn.Transformer.generate_square_subsequent_mask(x.size(1)), is_causal=True) + x
        x = self.expansion(x)
        x = self.var_layer(x, src_mask = T.nn.Transformer.generate_square_subsequent_mask(x.size(1)), is_causal=True)



        #x = T.sum(T.stack(T.split(x,temporal_division,dim=-1),dim=-1),dim=-1) ## chunk along the time dimension(adding a new dimension) then sum along the new dimension.
        mu =  T.chunk(x,2,dim=-1)[0]
        sigma = T.exp(T.chunk(x,2,dim=-1)[1])
        z = mu + (sigma*self.N.sample(mu.shape) if self.training else 0)
        self.kl = (sigma**2 + mu**2 - T.log(sigma) - 1/2).sum()

        ## This is a craZy way to implement variable compression lengths.
        ## I would much prefer a scan operator, but we aren't allowed nice things.

        reshaped = z[:,:,-1].unsqueeze(-1).expand(-1,-1,z.size(-2))

        reshaped = deNaNed(F.sigmoid(self.expansion_offset+z[:,:,-1].unsqueeze(-2).expand(-1, z.size(-2), -1)))+0.01 ## the +100 is a hack to ecnourage little compression at the beginning.
        sections = T.clamp(T.sum(T.tril(T.clamp(reshaped,max=1)),dim=-1)-1e-6, min=0)
        #sections = F.relu(T.where(sections[:,0].unsqueeze(-1)>=1, sections-1, sections))


        len_loss = softabs(T.amax(sections,dim=-1)-(x.size(1)/self.temporal_division), 1/3) ##punish lengths above len/temporal_div


        densified_z = T.zeros(  ( z.size(0), int(T.amax(T.ceil(sections)).item()), z.size(-1) ) , device=self.device)
        for i in range(x.size(0)):
            #print("??")
            ## Writen in pytorch 2.5 (apparently, this API may change in the future. )

            densified_z[i,T.floor(sections).int()] = T.index_reduce(z, dim=1, index=T.floor(sections[i]).int(), source=z, reduce='mean')
        ##
        return densified_z, len_loss, sections


class TVDT(T.nn.Module): # transformer variational encoder across time dimension.
    def __init__(self, data_width, emb_size=None,  temporal_division=8, layer_num=3, nheads = 6, device='cpu'):
        super().__init__()


        self.device=device
        self.nheads = nheads
        self.emb_size= emb_size if emb_size != None else data_width
        self.layers = [T.nn.TransformerEncoderLayer(data_width, nheads, dim_feedforward=data_width, batch_first=True, device=self.device )]*layer_num
        self.poscoder = PositionalEncoding(data_width, max_len=10_000, device=self.device)
        self.b_mix = BernoulliMix(0.5, device=self.device)


    def forward(self, z, skip_vec=None):

        z = self.poscoder(z)

        for i in range(len(self.layers)):
            layer=self.layers[i]

            #print("??2")
            if(i==len(self.layers)//2 and skip_vec!=None and self.training): ## mix in 'correct' answer at the middle layer
              z = layer(self.b_mix(z, skip_vec), src_mask = T.nn.Transformer.generate_square_subsequent_mask(z.size(1), device=self.device)) + z
            else:
              z = layer(z, src_mask = T.nn.Transformer.generate_square_subsequent_mask(z.size(1), device=self.device)) + z
        return z

class TVAT(T.nn.Module):
    def __init__(self, data_width, emb_size=None, temporal_division=8, layer_num=6, nheads=8, device='cpu'):
        super().__init__()
        self.device=device
        self.emb_size = emb_size if emb_size!=None else data_width
        self.encoder = TVET(data_width, emb_size, temporal_division, math.ceil(layer_num/2), nheads, device=self.device)
        self.decoder = TVDT(data_width, emb_size, temporal_division, math.floor(layer_num/2)+1, nheads, device=self.device)
        self.dropout = T.nn.Dropout()
        self.proj_up = T.nn.Linear(self.emb_size, data_width, device=self.device)
        self.kl=0



    def generate_mask(self, z, length, return_sizes=False):
      idxs=[0]*z.size(0)
      for i in range(z.size(0)):
          idxs[i] = T.repeat_interleave(T.clamp((1/(F.sigmoid(z[i,:,-1])+0.01)),max=length).int())[:length]

      #idxs = T.stack(idxs, dim=0)
      target_mask = T.full((z.size(0), length,length), -float('inf'), device=self.device)
      sizes=[None]*z.size(0)
      for i in range(z.size(0)): ###AAAAHHHH I HATE THISSS
        sizes[i] = T.histc(idxs[i].float(), T.clamp(T.amax(idxs[i]),min=1)   ).int().tolist()

        xpos = 0
        ypos = sizes[i][0]

        for j in range(len(sizes[i])-1):
          target_mask[i,xpos:xpos+sizes[i][j],ypos:]=0
          ypos+=sizes[i][j+1]
          xpos+=sizes[i][j]
      target_mask = target_mask.mT

      if(return_sizes):
        return target_mask, sizes
      else:
        return target_mask

    def encode(self, x, return_mask=False):
        result, len_loss, sections = self.encoder(x)
        self.kl = self.encoder.kl

        return result, len_loss, sections


    def decode(self, x, length, z=None):
        print(x.size(0),length, self.emb_size)
        expanded_z = T.zeros((x.size(0), length, self.emb_size), device=self.device)
        idxs=[0]*x.size(0)
        for i in range(x.size(0)):
            sizes = max_norm(T.clamp((1/(F.sigmoid(x[i,:,-1])+0.01)),max=x.size(1) ), x.size(1)*1.2)
            idxs[i] = T.repeat_interleave(sizes.int())[:length]
            expanded_z[i,:min(length,idxs[i].size(-1))] = x[i,idxs[i]][:length]
        '''
        #idxs = T.stack(idxs, dim=0)
        target_mask = T.full((x.size(0), x.size(-2),x.size(-2)), -float('inf'), device=self.device)

        for i in range(x.size(0)): ###AAAAHHHH I HATE THISSS
          #print(T.amax(idxs[i]))i
          sizes = T.histc(idxs[i].float(), T.clamp(T.amax(idxs[i]),min=1)    ).int().tolist()

          xpos = 0
          ypos = sizes[0]

          for j in range(len(sizes)-1):
            target_mask[i,xpos:xpos+sizes[j],ypos:]=0
            ypos+=sizes[j+1]
            xpos+=sizes[j]
        target_mask = target_mask.mT

        '''

        result = self.decoder(self.proj_up(expanded_z), z)
        mask = T.zeros((x.size(0), length), device=self.device).unsqueeze(-1)
        for i in range(x.size(0)):
            mask[i,:idxs[i].size(-1)]=1
        return mask * result




    def forward(self, x):
        z, len_loss, sections = self.encoder(x)
        self.kl = self.encoder.kl
        #print(T.mean(T.amax(sections,dim=-1)).item())

        expanded_z = T.zeros((x.size(0), x.size(1), self.emb_size), device=self.device)
        idxs=[0]*x.size(0)
        for i in range(x.size(0)):
            sizes = max_norm(T.clamp((1/(F.sigmoid(z[i,:,-1])+0.01)),max=x.size(1) ), x.size(1)*1.2)

            idxs[i] = T.repeat_interleave(sizes.int())[:x.size(1)]
            expanded_z[i,:min(x.size(1),idxs[i].size(-1))] = z[i,idxs[i]][:x.size(1)]
        '''
        #idxs = T.stack(idxs, dim=0)
        target_mask = T.full((x.size(0), x.size(-2),x.size(-2)), -float('inf'), device=self.device)

        for i in range(x.size(0)): ###AAAAHHHH I HATE THISSS
          sizes = T.histc(idxs[i].float(), T.clamp(T.amax(idxs[i]),min=1)   ).int().tolist()

          xpos = 0
          ypos = sizes[0]

          for j in range(len(sizes)-1):
            target_mask[i,xpos:xpos+sizes[j],ypos:]=0
            ypos+=sizes[j+1]
            xpos+=sizes[j]
        target_mask = target_mask.mT

        '''

        result = self.decoder(self.proj_up(expanded_z), self.decoder.poscoder(x))
        mask = T.zeros(x.size()[:-1], device=self.device).unsqueeze(-1)
        for i in range(x.size(0)):
            mask[i,:idxs[i].size(-1)]=1
        return mask * result, len_loss
'''
# modified from https://avandekleut.github.io/vae/
import math
import torch as T
import torch.nn.functional as F


class PositionalEncoding(T.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, device='cpu'):
        super().__init__()
        self.device=device
        self.dropout = T.nn.Dropout(p=dropout)

        position = T.arange(max_len, device=self.device).unsqueeze(1)
        div_term = T.exp(T.arange(0, d_model, 2, device=self.device) * (-math.log(10000.0) / d_model))
        pe = T.zeros((max_len, 1, d_model), device=self.device)
        pe[:, 0, 0::2] = T.sin(position * div_term)
        pe[:, 0, 1::2] = T.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x= x.transpose(0,1)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x).transpose(0,1)


def drop_tokens(x, probability):
    return x * T.where(T.rand(x.size()[:-1],device=x.device)>probability,0,1).unsqueeze(-1)

def direction(x):
    return T.round(T.clamp(x, min=-1, max=1))

def deNaNed(x):
    return T.where(T.isnan(x),0,x)

class TVET(T.nn.Module): # transformer variational encoder across time dimension.
    def __init__(self, data_width, emb_size=None,  temporal_division=8, layer_num=3, nheads = 6, device='cpu'):
        super().__init__()
        self.device=device
        self.emb_size = emb_size if emb_size!=None else data_width
        self.layers = [T.nn.TransformerEncoderLayer(data_width, nheads, dim_feedforward=data_width, batch_first=True, device=self.device )]*(layer_num-1)
        self.expansion = T.nn.Linear(data_width, self.emb_size*2, device=self.device)
        self.var_layer = T.nn.TransformerEncoderLayer(self.emb_size*2, nheads, dim_feedforward=data_width*2, batch_first=True, device=self.device)
        self.poscoder = PositionalEncoding(data_width, max_len=10_000, device=self.device)

        self.expansion_offset=0

        self.temporal_division = temporal_division


        self.N = T.distributions.Normal(0, 1)
        if(isinstance(device, str) and "cuda" in device):
            self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = self.poscoder(x)

        for layer in self.layers:
            x = layer(x, src_mask = T.nn.Transformer.generate_square_subsequent_mask(x.size(1), device=self.device), is_causal=True) + x
        x = self.expansion(x)
        x = self.var_layer(x, src_mask = T.nn.Transformer.generate_square_subsequent_mask(x.size(1),device=self.device), is_causal=True)



        #x = T.sum(T.stack(T.split(x,temporal_division,dim=-1),dim=-1),dim=-1) ## chunk along the time dimension(adding a new dimension) then sum along the new dimension.
        mu =  T.chunk(x,2,dim=-1)[0]
        sigma = T.exp(T.chunk(x,2,dim=-1)[1])
        z = mu + (sigma*self.N.sample(mu.shape) if self.training else 0)
        self.kl = (sigma**2 + mu**2 - T.log(sigma) - 1/2).sum()

        ## This is a craZy way to implement variable compression lengths.
        ## I would much prefer a scan operator, but we aren't allowed nice things.

        reshaped = z[:,:,-1].unsqueeze(-1).expand(-1,-1,z.size(-2))

        reshaped = deNaNed(F.sigmoid(self.expansion_offset+z[:,:,-1].unsqueeze(-2).expand(-1, z.size(-2), -1)))+0.01 ## the +100 is a hack to ecnourage little compression at the beginning.
        sections = T.clamp(T.sum(T.tril(T.clamp(reshaped,max=1)),dim=-1)-1e-6, min=0)
        #sections = F.relu(T.where(sections[:,0].unsqueeze(-1)>=1, sections-1, sections))


        len_loss = T.abs(T.amax(sections,dim=-1)-(x.size(1)/self.temporal_division)) ##punish lengths above len/temporal_div


        densified_z = T.zeros(  ( z.size(0), int(T.amax(T.ceil(sections)).item()), z.size(-1) ) , device=self.device)
        for i in range(x.size(0)):
            #print("??")
            ## Writen in pytorch 2.5 (apparently, this API may change in the future. )

            densified_z[i,T.floor(sections).int()] = T.index_reduce(z, dim=1, index=T.floor(sections[i]).int(), source=z, reduce='mean')
        ##
        return densified_z, len_loss, sections


class TVDT(T.nn.Module): # transformer variational encoder across time dimension.
    def __init__(self, data_width, emb_size=None,  temporal_division=8, layer_num=3, nheads = 6, device='cpu'):
        super().__init__()


        self.device=device
        self.nheads = nheads
        self.emb_size= emb_size if emb_size != None else data_width
        self.layers = [T.nn.TransformerDecoderLayer(data_width, nheads, dim_feedforward=data_width, batch_first=True, device=self.device )]*layer_num
        self.proj_up = T.nn.Linear(self.emb_size, data_width, device=self.device)
        self.poscoder = PositionalEncoding(data_width, max_len=10_000, device=self.device)


    def forward(self, z, target, target_mask):
        z = self.poscoder(self.proj_up(z))
        print(target_mask[0])
        for layer in self.layers:

            #print("??2")
            print("dec_z", z)
            z = layer(target, z, memory_mask = T.nn.Transformer.generate_square_subsequent_mask(z.size(1)-1, device=self.device), tgt_mask = T.repeat_interleave(target_mask, self.nheads,dim=0)) + z
        return z

class TVAT(T.nn.Module):
    def __init__(self, data_width, emb_size=None, temporal_division=8, layer_num=6, nheads=8, device='cpu'):
        super().__init__()
        self.device=device
        self.emb_size = emb_size if emb_size!=None else data_width
        self.encoder = TVET(data_width, emb_size, temporal_division, math.ceil(layer_num/2), nheads, device=self.device)
        self.decoder = TVDT(data_width, emb_size, temporal_division, math.floor(layer_num/2), nheads, device=self.device)
        self.kl=0


    def encode(self, x):
        result, len_loss, sections = self.encoder(x)
        self.kl = self.encoder.kl
        return result, len_loss, sections


    def decode(self, z, x):

        expanded_z = T.zeros((x.size(0), x.size(1), self.emb_size), device=self.device)
        idxs=[0]*x.size(0)
        for i in range(x.size(0)):
            idxs[i] = T.repeat_interleave(T.clamp((1/(F.sigmoid(z[i,:,-1])+0.01)),max=x.size(1) ).int())[:x.size(1)]
            expanded_z[i,:min(x.size(1),idxs[i].size(-1))] = z[i,idxs[i]][:x.size(1)]

        #idxs = T.stack(idxs, dim=0)
        target_mask = T.full((x.size(0), x.size(-2),x.size(-2)), -float('inf'), device=self.device)

        for i in range(x.size(0)): ###AAAAHHHH I HATE THISSS
          #print(T.amax(idxs[i]))i
          sizes = T.histc(idxs[i].float(), T.clamp(T.amax(idxs[i]),min=1)    ).int().tolist()
          
          xpos = 0
          ypos = sizes[0]

          for j in range(len(sizes)-1):
            target_mask[i,xpos:xpos+sizes[j],ypos:]=0
            ypos+=sizes[j+1]
            xpos+=sizes[j]

        target_mask = target_mask.mT


        result = self.decoder(expanded_z, x, target_mask) ## decoder should use teacher forcing.
        mask = T.zeros(x.size()[:-1], device=self.device).unsqueeze(-1)
        for i in range(x.size(0)):
            mask[i,:idxs[i].size(-1)]=1
        return mask * result




    def forward(self, x):
        z, len_loss, sections = self.encoder(x)
        self.kl = self.encoder.kl
        #print(T.mean(T.amax(sections,dim=-1)).item())

        expanded_z = T.zeros((x.size(0), x.size(1), self.emb_size), device=self.device)
        idxs=[0]*x.size(0)
        for i in range(x.size(0)):
            idxs[i] = T.repeat_interleave(T.clamp((1/(F.sigmoid(z[i,:,-1])+0.01)),max=x.size(1) ).int())[:x.size(1)]
            expanded_z[i,:min(x.size(1),idxs[i].size(-1))] = z[i,idxs[i]][:x.size(1)]

        #idxs = T.stack(idxs, dim=0)
        target_mask = T.full((x.size(0), x.size(-2),x.size(-2)), -float('inf'), device=self.device)

        for i in range(x.size(0)): ###AAAAHHHH I HATE THISSS
          sizes = T.histc(idxs[i].float(), T.clamp(T.amax(idxs[i]),min=1)   ).int().tolist()

          xpos = 0
          ypos = sizes[0]

          for j in range(len(sizes)-1):
            target_mask[i,xpos:xpos+sizes[j],ypos:]=0
            ypos+=sizes[j+1]
            xpos+=sizes[j]
        target_mask = target_mask.mT


        result = self.decoder(expanded_z, x, target_mask) ## decoder should use teacher forcing.
        mask = T.zeros(x.size()[:-1], device=self.device).unsqueeze(-1)
        for i in range(x.size(0)):
            mask[i,:idxs[i].size(-1)]=1
        return mask * result, len_loss

'''
