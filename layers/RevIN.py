# code from https://github.com/ts-kim/RevIN, with minor modifications

import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        x.to(torch.device('cuda:{}'.format(0)))
        assert x.device.type == 'cuda', 'RevIN.py -> RevIN -> forward -> x'
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        assert self.affine_weight.device == 'cuda:0', 'RevIN.py -> RevIN -> _init_params -> self.affine_weight'
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
        assert self.affine_bias.device == 'cuda:0', 'RevIN.py -> RevIN -> _init_params -> self.affine_bias'

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
            assert self.last.device == 'cuda:0', 'RevIN.py -> RevIN -> _get_statistics -> self.last'
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach().to(torch.device('cuda:{}'.format(0)))
            assert self.mean.device.type == 'cuda', 'RevIN.py -> RevIN -> _get_statistics -> self.mean'
            assert self.mean.device.index == 0, 'RevIN.py -> RevIN -> _get_statistics -> self.mean2'
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach().to(torch.device('cuda:{}'.format(0)))
        assert self.stdev.device.type == 'cuda', 'RevIN.py -> RevIN -> _get_statistics -> self.stdev'

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        x.to(torch.device('cuda:{}'.format(0)))
        assert x.device.type == 'cuda', 'RevIN.py -> RevIN -> _normalize -> x'
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        assert x.device.type == 'cuda', 'RevIN.py -> RevIN -> _denormalize -> x'
        return x
