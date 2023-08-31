import torch
import torch.nn as nn

from torch.nn.functional import softmax
from copy import deepcopy


class Embeddings(nn.Module):
    def __init__(self, input_size=2, max_length=100, dim_out=256):
        super().__init__()
        self.dim_out = dim_out
        # TODO: Try Embedding for 3-D inputs (Check for various ways of doing this)
        self.embbedL = nn.Embedding(input_size, dim_out)
        # TODO: Check for a better way to do positional embedding
        self.embbedP = nn.Embedding(max_length, dim_out)

    def forward(self,inputs):
        '''
        param @inputs: Tensor size (N,T) (batch, input_size)
        @rtype: Tensor size(N,T,H), H-embedding size
        '''
        pos = torch.arange(inputs.shape[1]).repeat(inputs.shape[0],1)
        return self.embbedL(inputs)+self.embbedP(pos)


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, model_dim, dropout=0.1):
        super().__init__()
        assert model_dim % heads == 0 
        self.dk = model_dim // heads
        self.heads = heads
        self.linear = [deepcopy(nn.Linear(model_dim,model_dim))
                        for _ in range(4)]
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self,query,key,value,mask=None):

        N = query.size(0)
        query,key,value = [linear(x).view(N,-1,self.heads,self.dk).transpose(1,2) 
                           for linear,x in zip(self.linear,(query,key,value))]
        scores = query@key.transpose(-2,-1)/(self.dk**.5)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill_(mask==0,value=-1e9)
        self.attn = softmax(scores,dim=-1)
        if self.dropout:
            self.attn = self.dropout(self.attn)
        input = self.attn@value
        input = input.transpose(1,2).contiguous().view(N,-1,self.heads *self.dk)
        return self.linear[-1](input)

class FeedForward(nn.Module):
    def __init__(self,model_dim,ff_dim,dropout=0.1):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(model_dim,ff_dim),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(ff_dim,model_dim)
                                )

    def forward(self,input):
        return self.ff(input)

class EncoderLayer(nn.Module):
    def __init__(self,model_dim, attn, ff, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.attn = attn
        self.ff = ff
        self.dropout  = nn.Dropout(dropout)
        self.LayerNorm =nn.LayerNorm(model_dim)
    def forward(self, input, mask):
        input+= self.dropout(self.attn(input,input,input,mask))
        out = self.LayerNorm(input)
        input+=self.dropout(self.ff(out))
        out = self.LayerNorm(input)

        return out

class Encoder(nn.Module):
    def __init__(self, Layer, n):
        super().__init__()
        self.layers = [deepcopy(Layer) 
                       for _ in range(n)]
        self.norm = nn.LayerNorm(Layer.model_dim)
    def forward(self,input,mask):
        for layer in self.layers:
            input = layer(input,mask)
        return self.norm(input)
    
class DecoderLayer(nn.Module):
    def __init__(self,model_dim,self_attn, src_attn,ff,dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.ff = ff
        self.LayerNorm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,input,mem,src_mask,tgt_mask):
        
        input+= self.dropout(self.self_attn(input,input,input,tgt_mask))
        out = self.LayerNorm(input)
        input+= self.dropout(self.src_attn(out,mem,mem,src_mask))
        out = self.LayerNorm(input)
        input+=self.dropout(self.ff(out))
        out = self.LayerNorm(input)

        return out

class Decoder(nn.Module):
    def __init__(self, Layer, n):
        super().__init__()
        self.layers = [deepcopy(Layer) 
                       for _ in range(n)]
        self.norm = nn.LayerNorm(Layer.model_dim)
    def forward(self,input,mem,src_mask,tgt_mask):
        for layer in self.layers:
            input = layer(input,mem,src_mask,tgt_mask)
        return self.norm(input)

class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder,src_emb,tgt_emb):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb

    def forward(self, src, tgt, src_mask, tgt_mask):
        out = self.encoder(self.src_emb(src), src_mask)
        out = self.decoder(self.tgt_emb(tgt), out, src_mask, tgt_mask)
        
        return out

class Generator(nn.Module):
    def __init__(self,model_dim,out_dim):
        super().__init__()
        self.linear = nn.Linear(model_dim, out_dim)
    
    def forward(self,input):
        return self.linear(input)
