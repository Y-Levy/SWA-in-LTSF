__all__ = ['PatchTST_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

#from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.RevIN import RevIN

# Cell
class PatchTST_backbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                 verbose:bool=False, **kwargs):
        
        super().__init__()

        # RevIn
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1
        
        # Backbone 
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
            assert self.head.device == 'cuda:0', 'PatchTST_backbone.py -> PatchTST_backbone -> __init__ -> if -> self.head'
        elif head_type == 'flatten': 
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
        
    
    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
        # norm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm').to(torch.device('cuda:{}'.format(0)))
            z = z.permute(0,2,1)
            
        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]
        
        # model
        z = self.backbone(z)                                                                # z: [bs x nvars x d_model x patch_num]
        z = self.head(z)                                                                    # z: [bs x nvars x target_window] 
        
        # denorm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        assert z.device.type == 'cuda', 'PatchTST_backbone.py -> PatchTST_backbone -> forward -> z'
        return z
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        assert nn.Sequential(nn.Dropout(dropout), nn.Conv1d(head_nf, vars, 1)).device == 'cuda:0', 'PatchTST_backbone.py -> PatchTST_backbone -> create_pretrain_head'
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        assert x.device.type == 'cuda', 'PatchTST_backbone.py -> Flatten_Head -> forward -> x'
        return x
        

class TSTiEncoder(nn.Module):  #i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        
        
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        
        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space

        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model).to(torch.device('cuda:{}'.format(0)))
        assert self.W_pos.device.type == 'cuda', 'PatchTST_backbone.py -> TSTiEncoder -> __init__ -> self.W_pos'
        assert self.W_pos.device.index == 0, 'PatchTST_backbone.py -> TSTiEncoder -> __init__ -> self.W_pos2'

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)

        
    def forward(self, x) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]
        
        x = x.double()
        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0,1,3,2).to(torch.device('cuda:{}'.format(0)))                                                # x: [bs x nvars x patch_num x patch_len]
        assert x.device.type == 'cuda', 'PatchTST_backbone.py -> TSTiEncoder -> forward -> x.permute(0,1,3,2)'
        self.W_P.to(torch.float64)
        x = self.W_P(x).to(torch.device('cuda:{}'.format(0)))                                                   # x: [bs x nvars x patch_num x d_model]
        assert x.device.type == 'cuda', 'PatchTST_backbone.py -> TSTiEncoder -> forward -> self.W_P(x)'

        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3])).to(torch.device('cuda:{}'.format(0)))      # u: [bs * nvars x patch_num x d_model]
        assert u.device.type == 'cuda', 'PatchTST_backbone.py -> TSTiEncoder -> forward -> u: torch.reshape()'
        u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x patch_num x d_model]

        # Encoder
        z = self.encoder(u)                                                      # z: [bs * nvars x patch_num x d_model]
        assert z.device.type == 'cuda', 'PatchTST_backbone.py -> TSTiEncoder -> forward -> z: self.encoder()'
        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))                # z: [bs x nvars x patch_num x d_model]
        assert z.device.type == 'cuda', 'PatchTST_backbone.py -> TSTiEncoder -> forward -> z: torch.reshape()'
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]
        assert z.device.type == 'cuda', 'PatchTST_backbone.py -> TSTiEncoder -> forward -> z'
        return z    

    
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            assert output.device.type == 'cuda', 'PatchTST_backbone.py -> TSTEncoder -> forward -> if -> output'
            assert scores.device.type == 'cuda', 'PatchTST_backbone.py -> TSTEncoder -> forward -> if -> scores'
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            assert output.device.type == 'cuda', 'PatchTST_backbone.py -> TSTEncoder -> forward -> else -> output'
            return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)
            assert self.norm_attn.device == 'cuda:0', 'PatchTST_layers.py -> TSTEncoderLayer -> __init__ -> Add & Norm -> else -> self.norm_attn'

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)
            assert self.norm_ffn.device == 'cuda:0', 'PatchTST_layers.py -> TSTEncoderLayer -> __init__ -> Add & Norm -> else -> self.norm_ffn'

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
            assert src.device.type == 'cuda', 'PatchTST_backbone.py -> TSTEncoderLayer -> forward -> Multi-Head attention sublayer ->  src'
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            assert src2.device.type == 'cuda', 'PatchTST_backbone.py -> TSTEncoderLayer -> forward -> Multi-Head attention -> if -> src2'
            assert attn.device.type == 'cuda', 'PatchTST_backbone.py -> TSTEncoderLayer -> forward -> Multi-Head attention -> if -> attn'
            assert scores.device.type == 'cuda', 'PatchTST_backbone.py -> TSTEncoderLayer -> forward -> Multi-Head attention -> if -> scores'
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            assert src2.device.type == 'cuda', 'PatchTST_backbone.py -> TSTEncoderLayer -> forward -> Multi-Head attention -> else -> src2'
            assert attn.device.type == 'cuda', 'PatchTST_backbone.py -> TSTEncoderLayer -> forward -> Multi-Head attention -> else -> attn'

        if self.store_attn:
            self.attn = attn
            assert self.attn.device.type == 'cuda', 'PatchTST_backbone.py -> TSTEncoderLayer -> forward -> if self.store_attn -> self.attn'
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        assert src.device.type == 'cuda', 'PatchTST_backbone.py -> TSTEncoderLayer -> forward -> Add & Norm -> src'
        if not self.pre_norm:
            src = src.to(torch.float32)
            src = self.norm_attn(src)
            assert src.device.type == 'cuda', 'PatchTST_backbone.py -> TSTEncoderLayer -> forward -> Add & Norm -> if not -> src'

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
            assert src.device.type == 'cuda', 'PatchTST_backbone.py -> TSTEncoderLayer -> forward -> Feed-forward sublayer -> if -> src'
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        assert src2.device.type == 'cuda', 'PatchTST_backbone.py -> TSTEncoderLayer -> forward -> Position-wise Feed-Forward -> src2'
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        assert src.device.type == 'cuda', 'PatchTST_backbone.py -> TSTEncoderLayer -> forward -> Add & Norm -> src'
        if not self.pre_norm:
            src = self.norm_ffn(src)
            assert src.device.type == 'cuda', 'PatchTST_backbone.py -> TSTEncoderLayer -> forward -> Feed-forward sublayer -> Add & Norm -> if not -> src'

        if self.res_attention:
            assert src.device.type == 'cuda', 'PatchTST_backbone.py -> TSTEncoderLayer -> forward -> if self.res_attention -> src'
            assert scores.device.type == 'cuda', 'PatchTST_backbone.py -> TSTEncoderLayer -> forward -> if self.res_attention -> scores'
            return src, scores
        else:
            assert src.device == 'cuda:0', 'PatchTST_layers.py -> TSTEncoderLayer -> forward -> else self.res_attention -> src'
            return src


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias).to(torch.device('cuda:{}'.format(0)))
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias).to(torch.device('cuda:{}'.format(0)))
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias).to(torch.device('cuda:{}'.format(0)))

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q.to(torch.device('cuda:{}'.format(0))).to(torch.float32)
        assert K.device.type == 'cuda', 'PatchTST_backbone.py -> _MultiheadAttention -> forward -> K'
        if V is None: V = Q.to(torch.device('cuda:{}'.format(0))).to(torch.float32)
        assert V.device.type == 'cuda', 'PatchTST_backbone.py -> _MultiheadAttention -> forward -> V'
        Q = Q.to(torch.float32)
        K = K.to(torch.float32)
        V = V.to(torch.float32)

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2).to(torch.device('cuda:{}'.format(0)))       # q_s    : [bs x n_heads x max_q_len x d_k]
        assert q_s.device.type == 'cuda', 'PatchTST_backbone.py -> _MultiheadAttention -> forward -> q_s'
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1).to(torch.device('cuda:{}'.format(0)))     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        assert k_s.device.type == 'cuda', 'PatchTST_backbone.py -> _MultiheadAttention -> forward -> k_s'
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2).to(torch.device('cuda:{}'.format(0)))       # v_s    : [bs x n_heads x q_len x d_v]
        assert v_s.device.type == 'cuda', 'PatchTST_backbone.py -> _MultiheadAttention -> forward -> v_s'

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            assert output.device.type == 'cuda', 'PatchTST_backbone.py -> _MultiheadAttention -> forward -> Apply Scaled Dot-Product.. -> if -> output'
            assert attn_weights.device.type == 'cuda', 'PatchTST_backbone.py -> _MultiheadAttention -> forward -> Apply Scaled Dot-Product.. -> if -> attn_weights'
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            assert output.device.type == 'cuda', 'PatchTST_backbone.py -> _MultiheadAttention -> forward -> Apply Scaled Dot-Product.. -> else -> output'
            assert attn_weights.device.type == 'cuda', 'PatchTST_backbone.py -> _MultiheadAttention -> forward -> Apply Scaled Dot-Product.. -> else -> attn_weights'

        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v).to(torch.device('cuda:{}'.format(0))) # output: [bs x q_len x n_heads * d_v]
        assert output.device.type == 'cuda', 'PatchTST_backbone.py -> _MultiheadAttention -> forward -> output: back to the original inputs dimensions -> output'
        output = self.to_out(output).to(torch.device('cuda:{}'.format(0)))
        assert output.device.type == 'cuda', 'PatchTST_backbone.py -> _MultiheadAttention -> forward -> output: back to the original inputs dimensions -> self.to_out(output)'

        if self.res_attention:
            attn_weights.to(torch.device('cuda:{}'.format(0)))
            assert output.device.type == 'cuda', 'PatchTST_backbone.py -> _MultiheadAttention -> forward -> if self.res_attention -> output'
            assert attn_weights.device.type == 'cuda', 'PatchTST_backbone.py -> _MultiheadAttention -> forward -> if self.res_attention -> attn_weights'
            assert attn_scores.device.type == 'cuda', 'PatchTST_backbone.py -> _MultiheadAttention -> forward -> if self.res_attention -> attn_scores'
            return output, attn_weights, attn_scores
        else:
            assert output.device == 'cuda:0', 'PatchTST_layers.py -> _MultiheadAttention -> forward -> else self.res_attention -> output'
            assert attn_weights.device == 'cuda:0', 'PatchTST_layers.py -> _MultiheadAttention -> forward -> else self.res_attention -> attn_weights'
            return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa).to(torch.device('cuda:{}'.format(0)))
        assert self.scale.device.type == 'cuda', 'PatchTST_backbone.py -> _ScaledDotProductAttention -> __init__ -> self.scale'
        assert self.scale.device.index == 0, 'PatchTST_backbone.py -> _ScaledDotProductAttention -> __init__ -> self.scale2'
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]
        attn_scores.to(torch.device('cuda:{}'.format(0)))
        assert attn_scores.device.type == 'cuda', 'PatchTST_layers.py -> _ScaledDotProductAttention -> forward -> attn_scores'

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask
            assert attn_scores.device == 'cuda:0', 'PatchTST_layers.py -> _ScaledDotProductAttention -> forward -> if attn_mask -> attn_scores'

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)
            assert attn_scores.masked_fill_ == 'cuda:0', 'PatchTST_backbone.py -> _ScaledDotProductAttention -> forward -> if key_padding_mask -> attn_scores.masked_fill_'

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1).to(torch.device('cuda:{}'.format(0)))                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        assert attn_weights.device.type == 'cuda', 'PatchTST_backbone.py -> _ScaledDotProductAttention -> forward -> attn_weights'
        attn_weights = self.attn_dropout(attn_weights).to(torch.device('cuda:{}'.format(0)))
        assert attn_weights.device.type == 'cuda', 'PatchTST_backbone.py -> _ScaledDotProductAttention -> forward -> self.attn_dropout(attn_weights)'

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v).to(torch.device('cuda:{}'.format(0)))                        # output: [bs x n_heads x max_q_len x d_v]
        assert output.device.type == 'cuda', 'PatchTST_backbone.py -> _ScaledDotProductAttention -> forward -> output'

        if self.res_attention:
            assert output.device.type == 'cuda', 'PatchTST_layers.py -> _ScaledDotProductAttention -> forward -> if self.res_attention -> output'
            assert attn_weights.device.type == 'cuda', 'PatchTST_layers.py -> _ScaledDotProductAttention -> forward -> if self.res_attention -> attn_weights'
            assert attn_scores.device.type == 'cuda', 'PatchTST_layers.py -> _ScaledDotProductAttention -> forward -> if self.res_attention -> attn_scores'
            return output, attn_weights, attn_scores
        else:
            assert output.device == 'cuda:0', 'PatchTST_layers.py -> _ScaledDotProductAttention -> forward -> else self.res_attention -> output'
            assert attn_weights.device == 'cuda:0', 'PatchTST_layers.py -> _ScaledDotProductAttention -> forward -> else self.res_attention -> attn_weights'
            return output, attn_weights

