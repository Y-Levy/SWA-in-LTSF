__all__ = ['Exp_Basic']

import neptune
from torch.optim.swa_utils import AveragedModel
import os
import torch
import random
import numpy as np
from models import autoformer, fedformer, patchtst
from models.nbeats_models import generic, interpretable


class Exp_Basic(object):
    def __init__(self, args):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.args = args
        self.model_dict = {
            'Autoformer': autoformer,
            'FEDformer': fedformer,
            'PatchTST': patchtst,
        }
        self.device = self._acquire_device()
        self.model = self._build_model(args).to(self.device)
        self.use_swa = 1 if args.swa_start <= args.train_epochs else 0
        self.eps = 0.1
        if not self.use_swa:
            print(f'Running regular {args.model}.\n')
        elif self.args.custom_averaging:
            print("Running custom averaging on SWA - exponential moving average.\n")
            ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged : self.eps * averaged_model_parameter + (1-self.eps) * model_parameter
            self.swa_model = torch.optim.swa_utils.AveragedModel(self.model, avg_fn=ema_avg).to(self.device)
        else:
            print("Running equal averaging on SWA.\n")
            self.swa_model = AveragedModel(self.model).to(self.device)

    def _build_model(self, args):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass


