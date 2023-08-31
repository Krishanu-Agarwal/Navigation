import torch
import torch.nn as nn

from torch.nn.functional import softmax
from copy import deepcopy

from Modules import *

class Transformer(nn.Module):
    def __init__(self,
                 max_inp_dim = 100,
                 heads = 8,
                 model_dim =256,
                 ff_dim = 1024,
                 out_dim = 2,
                 dropout = 0.1,
                 Nx = 4):
        super().__init__()
        #Embeddings
        self.src_emb = Embeddings(max_inp_dim, max_inp_dim, model_dim)
        self.tgt_emb = Embeddings(max_inp_dim, max_inp_dim, model_dim)
        #Multihead attention
        self.en_attn = MultiHeadAttention(heads, model_dim, dropout)
        self.de_self_attn = MultiHeadAttention(heads, model_dim, dropout)
        self.de_src_attn = MultiHeadAttention(heads, model_dim, dropout)
        #Feedforward layers
        self.en_ff = FeedForward(model_dim, ff_dim, dropout)
        self.de_ff = FeedForward(model_dim, ff_dim, dropout)
        #Encoder
        self.encoder = Encoder(EncoderLayer(model_dim, self.en_attn, self.en_ff, dropout), Nx)
        #Decoder
        self.decoder = Decoder(DecoderLayer(model_dim, self.de_self_attn, self.de_src_attn, self.de_ff,dropout), Nx)
        #Combinig Encoder and Decoder
        self.en_de =EncoderDecoder(self.encoder, self.decoder, self.src_emb, self.tgt_emb)

        #Generator
        self.Gen = Generator(model_dim, out_dim)
    
    def forward(self, src_data, tgt_data, src_mask, tgt_mask):
        return self.Gen(self.en_de(src_data,tgt_data,src_mask,tgt_mask))