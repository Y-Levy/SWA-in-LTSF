__all__ = ['Exp_Long_Term_Forecast']

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import (EarlyStopping, adjust_learning_rate, visual, test_params_flop, adjust_learning_rate_TST,
                         get_epochs_to_aggregate, get_weights_dict, weight_comparison_exp, get_weights_and_grads,
                         compute_norms, read_grads_file, MyTupleWrapper, ensemble_P_i)
from utils.metrics import metric
from utils.density_plot import get_esd_plot, density_generate, gaussian, MyPyhessian, list_eigenvectors
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.optim.swa_utils import SWALR
from torch.nn.utils import (parameters_to_vector as Params2Vec, vector_to_parameters as Vec2Params)
from pyhessian import hessian
import shutil
import datetime
import copy
from tqdm import tqdm
import random
import glob
import csv
import os
import time
import warnings
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self, args):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, swa_flag=False):
        vali_model = self.swa_model if swa_flag else self.model
        vali_model.to(self.device)

        total_mae = []
        total_mse = []
        total_rmse = []
        total_mape = []
        total_mspe = []
        total_loss = []
        vali_model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(vali_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.model in ['NBEATS-G', 'NBEATS-I']:
                    batch_x = batch_x.squeeze(-1)
                    batch_y = batch_y[:, -self.args.pred_len:].squeeze(-1)
                    input_mask = torch.ones(batch_x.shape).to(self.device)
                    outputs = vali_model(batch_x, input_mask)
                else:
                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.model in ['PatchTST', 'DLinear']:
                                outputs = vali_model(batch_x)
                            else:
                                my_tuple = MyTupleWrapper(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                if self.args.output_attention:
                                    outputs = vali_model(my_tuple)[0]
                                else:
                                    outputs = vali_model(my_tuple)
                    else:
                        if self.args.model in ['PatchTST', 'DLinear']:
                            outputs = vali_model(batch_x)
                        else:
                            my_tuple = MyTupleWrapper(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            if self.args.output_attention:
                                outputs = vali_model(my_tuple)[0]
                            else:
                                outputs = vali_model(my_tuple)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                preds = np.array(pred)
                trues = np.array(true)
                mae, mse, rmse, mape, mspe = metric(preds, trues)

                total_loss.append(loss)
                total_mae.append(mae)
                total_mse.append(mse)
                total_rmse.append(rmse)
                total_mape.append(mape)
                total_mspe.append(mspe)

        total_loss = np.average(total_loss)
        total_mae = np.average(total_mae)
        total_mse = np.average(total_mse)
        total_rmse = np.average(total_rmse)
        total_mape = np.average(total_mape)
        total_mspe = np.average(total_mspe)

        vali_model.train()
        return total_loss, total_mae, total_mse, total_rmse, total_mape, total_mspe

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and self.args.ii == 0:
            os.makedirs(path)

        if self.args.record:
            if self.use_swa:
                global path_P_i_swa, path_D_i_swa

                path_P_i_swa = os.path.join(self.args.checkpoints, setting + f'/P_i_swa/')
                if not os.path.exists(path_P_i_swa) and self.args.ii == 0:
                    os.makedirs(path_P_i_swa)

                path_D_i_swa = os.path.join(self.args.checkpoints, setting + f'/D_i_swa/')
                if not os.path.exists(path_D_i_swa) and self.args.ii == 0:
                    os.makedirs(path_D_i_swa)

                swa_epoch_derivatives = {}
            else:
                global path_W_i_original, path_G_i_original

                path_W_i_original = os.path.join(self.args.checkpoints, setting + '/W_i_original/') #if self.use_swa else ...
                if not os.path.exists(path_W_i_original) and self.args.ii == 0:
                    os.makedirs(path_W_i_original)
                    print('\nrunning W_i for the first time.')

                path_G_i_original = os.path.join(self.args.checkpoints, setting + '/G_i_original/')
                if not os.path.exists(path_G_i_original) and self.args.ii == 0:
                    os.makedirs(path_G_i_original)

                epoch_grads = {}

        # weights, grads = get_weights_and_grads(self.model)
        # print(f'model at start of train:\nweights vec: {weights[:10:]}\ngradients vec: {grads[:10]}\n')
        time_now = time.time()

        train_steps = len(train_loader)
        #TODO: take pic. of ESD
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        swa_start = self.args.swa_start
        swa_scheduler = SWALR(model_optim, anneal_strategy=self.args.anneal_strategy, anneal_epochs=self.args.anneal_epochs, swa_lr=self.args.swa_lr)

        if self.args.model == 'PatchTST':
            scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                                steps_per_epoch = train_steps,
                                                pct_start = self.args.pct_start,
                                                epochs = self.args.train_epochs,
                                                max_lr = self.args.learning_rate)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        step = 0
        for epoch in tqdm(range(1, self.args.train_epochs+1), desc='Epoch'):
            if (self.args.model == 'PatchTST' and epoch > 100) or (
                self.args.model != 'PatchTST' and epoch > 10):
                self.args.record = False

            # if self.use_swa:
            #     # if use_swa, we have 2 sets of weights (grads): model and swa_model
            #     weights, grads = get_weights_and_grads(self.model)
            #     print(f'model at start of train loop:\nweights vec: {weights[:10]}\ngradients vec: {grads[:10]}\n')
            #     swa_params, swa_derives = get_weights_and_grads(self.swa_model)
            #     print(f'swa_model at start of train loop:\nswa_params vec: {swa_params[:10:]}\nswa_derives vec: {swa_derives[:10]}\n')
            # else:
            #     weights, grads = get_weights_and_grads(self.model)
            #     print(f'model at start of train loop:\nweights vec: {weights[:10]}\ngradients vec: {grads[:10]}\n')

            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(train_loader, desc='Iter')):
                iter_count += 1
                step += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model in ['PatchTST', 'DLinear']:
                            outputs = self.model(batch_x)
                        else:
                            my_tuple = MyTupleWrapper(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            if self.args.output_attention:
                                outputs = self.model(my_tuple)[0]
                            else:
                                outputs = self.model(my_tuple)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.model in ['PatchTST', 'DLinear']:
                        outputs = self.model(batch_x)
                    else:
                        my_tuple = MyTupleWrapper(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        if self.args.output_attention:
                            outputs = self.model(my_tuple)[0]
                        else:
                            outputs = self.model(my_tuple)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward(retain_graph=True)
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward(retain_graph=True)
                    model_optim.step()

                if self.args.lradj == 'TST':
                    # original run
                    if epoch < swa_start:
                        my_lr = adjust_learning_rate_TST(model_optim, scheduler, epoch, self.args)
                        scheduler.step()
                    # SWA run
                    else:
                        my_lr = swa_scheduler.optimizer.param_groups[0]['lr']
                        swa_scheduler.step()
                    print(f'lr: {my_lr}')
                    if self.args.use_neptune:
                        self.args.neptune_run["scheduler/lr"].append(my_lr, step=step)


            print("Epoch: {} cost time: {}".format(epoch, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            if epoch >= swa_start:
                # record only for swa_model since use_swa=True
                print(f'\n\t>>>>>>>>>>>>>>>>> SWA <<<<<<<<<<<<<<<\nSWA step {epoch-swa_start+1}:  Updating swa_model ...')
                print('(Note this is still the regular model)') if epoch == swa_start else ...

                # 2 sets of weights (grads) - model and swa_model - what happens before swa update?
                # params, derives = get_weights_and_grads(self.model)
                # print(f'P_i before swa update:\nparams vec: {params[:10]}\nD_i vec: {grads[:10]}\n')
                # swa_params, swa_derives = get_weights_and_grads(self.swa_model)
                # print(f'P_swa before swa update:\nswa_params vec: {swa_params[:10]}\nD_swa vec: {swa_derives[:10]}\n')

                self.swa_model.update_parameters(self.model)

                # 2 sets of weights (grads) - model and swa_model - what happens after swa update?
                params, derives = get_weights_and_grads(self.model)
                print(f'P_i after swa update:\nparams vec: {params[:10]}\nD_i vec: {derives[:10]}\n')
                swa_params, swa_derives = get_weights_and_grads(self.swa_model)
                print(f'P_swa after swa update:\nswa_params vec: {swa_params[:10]}\nD_swa vec: {swa_derives[:10]}\n')

                # ==============================================================================
                if self.args.record:
                    # swa_model is just a place-holder of the running average so grad=False, grads of P_i building blocks (i.e. self.model) - are saved per epoch instead
                    swa_epoch_derivatives[f'epoch_{epoch}'] = derives
                    # save swa_model weights of the current epoch
                    torch.save(self.model.state_dict(), path_P_i_swa + 'epoch_{}.pth'.format(epoch))
                # ===============================================================================

                if self.args.use_neptune:
                    # Drives norm of swa_model, per epoch
                    swa_derives_l1_norm, swa_derives_l2_norm, swa_derives_max_norm = compute_norms(derives)
                    self.args.neptune_run[f'iter{self.args.ii}/swa_derives/l1_norm'].append(swa_derives_l1_norm, step=epoch)
                    self.args.neptune_run[f'iter{self.args.ii}/swa_derives/l2_norm'].append(swa_derives_l2_norm, step=epoch)
                    self.args.neptune_run[f'iter{self.args.ii}/swa_derives/max_norm'].append(swa_derives_max_norm, step=epoch)
                    # Params norm of swa_model, per epoch
                    swa_params_l1_norm, swa_params_l2_norm, swa_params_max_norm = compute_norms(swa_params)
                    self.args.neptune_run[f'iter{self.args.ii}/swa_weights/l1_norm'].append(swa_params_l1_norm, step=epoch)
                    self.args.neptune_run[f'iter{self.args.ii}/swa_weights/l2_norm'].append(swa_params_l2_norm, step=epoch)
                    self.args.neptune_run[f'iter{self.args.ii}/swa_weights/max_norm'].append(swa_params_max_norm, step=epoch)
                    del swa_params, swa_derives, params, derives

                    # if self.args.ii == 0:
                    #     self.args.neptune_run[f'model_checkpoints/SWA_{epoch}'].upload(path_W_swa + f'epoch_{epoch}.pth')
                    #     self.args.neptune_run[f'model_checkpoints/W_{epoch}_swa'].upload(path_W_j_swa + f'epoch_{epoch}.pth')

                vali_loss_swa = self.vali(vali_data, vali_loader, criterion, swa_flag=True)
                test_loss_swa = self.vali(test_data, test_loader, criterion, swa_flag=True)

                print("Epoch: {0}, Steps: {1} | Train SWA Loss: {2:.7f} Vali SWA Loss: {3:.7f} Test SWA Loss: {4:.7f}".format(
                       epoch, train_steps, train_loss, vali_loss_swa[0], test_loss_swa[0]))
                print(f'\nVali SWA MAE: {vali_loss_swa[1]:.7f} Test SWA MAE: {test_loss_swa[1]:.7f}')
                print(f'Vali SWA MSE: {vali_loss_swa[2]:.7f} Test SWA MSE: {test_loss_swa[2]:.7f}')
                print(f'Vali SWA RMSE: {vali_loss_swa[3]:.7f} Test SWA RMSE: {test_loss_swa[3]:.7f}')
                print(f'Vali SWA MAPE: {vali_loss_swa[4]:.7f} Test SWA MAPE: {test_loss_swa[4]:.7f}')
                print(f'Vali SWA MSPE: {vali_loss_swa[5]:.7f} Test SWA MSPE: {test_loss_swa[5]:.7f}\n')

                if self.args.use_neptune:
                    self.args.neptune_run[f'iter{self.args.ii}/train/loss'].append(train_loss, step=epoch) #self.args.neptune_run["train/swa_loss"].log(train_loss)
                    self.args.neptune_run[f'iter{self.args.ii}/validation/loss'].append(vali_loss_swa[0], step=epoch) #self.args.neptune_run["validation/swa_loss"].log(vali_loss_swa[0])
                    self.args.neptune_run[f'iter{self.args.ii}/test/loss'].append(test_loss_swa[0], step=epoch) #self.args.neptune_run["test/swa_loss"].log(test_loss_swa[0])
                    self.args.neptune_run[f'iter{self.args.ii}/validation/mae'].append(vali_loss_swa[1], step=epoch) #self.args.neptune_run["validation/swa_mae"].log(vali_loss_swa[1])
                    self.args.neptune_run[f'iter{self.args.ii}/test/mae'].append(test_loss_swa[1], step=epoch) #self.args.neptune_run["test/swa_mae"].log(test_loss_swa[1])
                    self.args.neptune_run[f'iter{self.args.ii}/validation/mse'].append(vali_loss_swa[2], step=epoch) #self.args.neptune_run["validation/swa_mse"].log(vali_loss_swa[2])
                    self.args.neptune_run[f'iter{self.args.ii}/test/mse'].append(test_loss_swa[2], step=epoch) #self.args.neptune_run["test/swa_mse"].log(test_loss_swa[2])
                    self.args.neptune_run[f'iter{self.args.ii}/validation/rmse'].append(vali_loss_swa[3], step=epoch) #self.args.neptune_run["validation/swa_rmse"].log(vali_loss_swa[3])
                    self.args.neptune_run[f'iter{self.args.ii}/test/rmse'].append(test_loss_swa[3], step=epoch) #self.args.neptune_run["test/swa_rmse"].log(test_loss_swa[3])
                    self.args.neptune_run[f'iter{self.args.ii}/validation/mape'].append(vali_loss_swa[4], step=epoch) #self.args.neptune_run["validation/swa_mape"].log(vali_loss_swa[4])
                    self.args.neptune_run[f'iter{self.args.ii}/test/mape'].append(test_loss_swa[4], step=epoch) #self.args.neptune_run["test/swa_mape"].log(test_loss_swa[4])
                    self.args.neptune_run[f'iter{self.args.ii}/validation/mspe'].append(vali_loss_swa[5], step=epoch) #self.args.neptune_run["validation/swa_mspe"].log(vali_loss_swa[5])
                    self.args.neptune_run[f'iter{self.args.ii}/test/mspe'].append(test_loss_swa[5], step=epoch) #self.args.neptune_run["test/swa_mspe"].log(test_loss_swa[5])

                if self.args.lradj != 'TST':
                    swa_scheduler.step()
                    my_lr = swa_scheduler.optimizer.param_groups[0]['lr']
                    print(f'Updating learning rate: {my_lr}')
                    if self.args.use_neptune:
                        self.args.neptune_run["scheduler/lr"].append(my_lr, step=epoch) # self.args.neptune_run["scheduler/swa_lr"].append(my_lr)

            else:
                # might be original baseline run, or swa run but epoch < swa_start - record for both models
                vali_loss = self.vali(vali_data, vali_loader, criterion, swa_flag=False)
                test_loss = self.vali(test_data, test_loader, criterion, swa_flag=False)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                      epoch, train_steps, train_loss, vali_loss[0], test_loss[0]))
                print(f'\nvali_mae: {vali_loss[1]:.7f}, test_mae: {test_loss[1]:.7f}')
                print(f'vali_mse: {vali_loss[2]:.7f}, test_mse: {test_loss[2]:.7f}')
                print(f'vali_rmse: {vali_loss[3]:.7f}, test_rmse: {test_loss[3]:.7f}')
                print(f'vali_mape: {vali_loss[4]:.7f}, test_mape: {test_loss[4]:.7f}')
                print(f'vali_mspe: {vali_loss[5]:.7f}, test_mspe: {test_loss[5]:.7f}\n')

                if self.args.use_neptune:
                    self.args.neptune_run[f'iter{self.args.ii}/train/loss'].append(train_loss, step=epoch)
                    self.args.neptune_run[f'iter{self.args.ii}/validation/loss'].append(vali_loss[0], step=epoch)
                    self.args.neptune_run[f'iter{self.args.ii}/test/loss'].append(test_loss[0], step=epoch)
                    self.args.neptune_run[f'iter{self.args.ii}/validation/mae'].append(vali_loss[1], step=epoch)
                    self.args.neptune_run[f'iter{self.args.ii}/test/mae'].append(test_loss[1], step=epoch)
                    self.args.neptune_run[f'iter{self.args.ii}/validation/mse'].append(vali_loss[2], step=epoch)
                    self.args.neptune_run[f'iter{self.args.ii}/test/mse'].append(test_loss[2], step=epoch)
                    self.args.neptune_run[f'iter{self.args.ii}/validation/rmse'].append(vali_loss[3], step=epoch)
                    self.args.neptune_run[f'iter{self.args.ii}/test/rmse'].append(test_loss[3], step=epoch)
                    self.args.neptune_run[f'iter{self.args.ii}/validation/mape'].append(vali_loss[4], step=epoch)
                    self.args.neptune_run[f'iter{self.args.ii}/test/mape'].append(test_loss[4], step=epoch)
                    self.args.neptune_run[f'iter{self.args.ii}/validation/mspe'].append(vali_loss[5], step=epoch)
                    self.args.neptune_run[f'iter{self.args.ii}/test/mspe'].append(test_loss[5], step=epoch)

                global best_epoch
                best_epoch = early_stopping(vali_loss[0], self.model, path, epoch)
                print(f'All parameters: {Params2Vec(self.model.parameters())}')
                torch.save(self.model.state_dict(), os.path.join('./checkpoints/' + setting, f'best_epoch.pth')) if best_epoch == epoch else ...

                # =============================== PyHessian =============================================
                # hessian_comp = MyPyhessian(self.model, criterion=criterion, data=(my_tuple, batch_y), cuda=True)
                # for _ in range(1):
                #     tol, top_n, maxIter = 1e-6, 1, int(1e2)
                #
                #     top_eigenvalues, top_eigenvectors = hessian_comp.eigenvalues(top_n=top_n, tol=tol, maxIter=maxIter)
                #     print(f'top {len(top_eigenvalues)} eigenvalues are: {top_eigenvalues}\n')
                #     print(f'Sum eigenvalues, {_}th try: {sum(top_eigenvalues)}')
                #     top_eigenvectors = list_eigenvectors(top_eigenvectors, top_n=top_n)
                #     print('Checking the norm of all eigenvectors is 1...')
                #     for j in range(top_n):
                #         print(f'l2-norm of eigenvector {j}: {torch.norm(top_eigenvectors[j])}')
                #
                #     trace = hessian_comp.trace(tol=tol, maxIter=maxIter)
                #     trace = sum(trace) / len(trace)
                #     print(f'Trace, {_}th try: {trace}\n')
                #
                # density_eigen, density_weight = hessian_comp.density()
                # figure = get_esd_plot(density_eigen, density_weight)
                #
                # if self.args.use_neptune:
                #     self.args.neptune_run[f'loss_landscape/Ensemble Sizes'].upload(figure)
                #     self.args.neptune_run[f'PyHessian/Lipschitz constant of the gradient'].append(top_eigenvalues[0], step=epoch)
                #     self.args.neptune_run[f'PyHessian/Trace'].append(trace, step=epoch)
                # =============================== PyHessian =============================================

                if self.use_swa:
                    ...
                    # swa_model before averaging starts
                    # weights, grads = get_weights_and_grads(self.model)
                    # print(f'model weights before regular break:\nweights vec: {weights[:10]}\ngradients vec: {grads[:10]}\n')
                    # swa_params, swa_derives = get_weights_and_grads(self.swa_model)
                    # print(f'swa_model params before regular break:\nswa_params vec: {swa_params[:10]}\nswa_derives vec: {swa_derives[:10]}\n')
                else:
                    # original baseline run
                    weights, grads = get_weights_and_grads(self.model)
                    print(f'model weights before regular break:\nweights vec: {weights[:10]}\ngradients vec: {grads[:10]}\n')

                    # ===========================================================================================
                    if self.args.record:
                        min_s = self.args.train_epochs
                        min_s_path = ''

                        for folder in os.listdir(self.args.checkpoints):
                            # check only SWA folders
                            if not folder.endswith('test') and "_".join(folder.split("_")[:-5])==setting:
                                s = int(folder.split("_")[-5].split("s")[1])
                                if s < min_s:
                                    min_s = s
                                    min_s_path = folder
                        # during original baseline run, check which epochs are saved in the entire swa_grads.csv file
                        df = pd.read_csv(os.path.join(self.args.checkpoints, min_s_path+'/D_i_swa', 'entire_swa_grads.csv'))
                        # save model grads/weights of the current epoch
                        if epoch >= min_s:
                            # to epochs_grads dict, only if epoch is in the entire swa_grads.csv file
                            epoch_grads[f'epoch_{epoch}'] = grads
                            # to epochs weights path, only if epoch is in the entire swa_grads.csv file
                            torch.save(self.model.state_dict(), path_W_i_original + 'epoch_{}.pth'.format(epoch))
                        del weights, grads
                    # ============================================================================================

                    if self.args.use_neptune:
                        # Grads norm of original_model, per epoch
                        grads_l1_norm, grads_l2_norm, grads_max_norm = compute_norms(grads)
                        self.args.neptune_run[f'iter{self.args.ii}/gradients/l1_norm'].append(grads_l1_norm, step=epoch)
                        self.args.neptune_run[f'iter{self.args.ii}/gradients/l2_norm'].append(grads_l2_norm, step=epoch)
                        self.args.neptune_run[f'iter{self.args.ii}/gradients/max_norm'].append(grads_max_norm, step=epoch)
                        # Weights norm of original_model, per epoch
                        weights_l1_norm, weights_l2_norm, weights_max_norm = compute_norms(weights)
                        self.args.neptune_run[f'iter{self.args.ii}/weights/l1_norm'].append(weights_l1_norm, step=epoch)
                        self.args.neptune_run[f'iter{self.args.ii}/weights/l2_norm'].append(weights_l2_norm, step=epoch)
                        self.args.neptune_run[f'iter{self.args.ii}/weights/max_norm'].append(weights_max_norm, step=epoch)

                if early_stopping.early_stop:
                    #torch.save(self.model.state_dict(), os.path.join('./checkpoints/' + setting, 'checkpoint.pth')) #torch.save(self.model.state_dict(), path, 'checkpoint.pth')
                    print("Early stopping")
                    break

                if self.args.lradj != 'TST':
                    my_lr = adjust_learning_rate(model_optim, epoch, self.args)
                else:
                    print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
                    print(f'new lr is: {my_lr}')
                if self.args.use_neptune:
                    self.args.neptune_run["scheduler/lr"].append(my_lr)

        # hessian = torch.autograd.functional.hessian(criterion, (outputs, batch_y))
        # fisher_info_matrix = -torch.autograd.functional.hessian(criterion, (outputs, batch_y))/len(weights)

        # END OF TRAINING
        if self.use_swa:
            # swa_model is now done training
            torch.optim.swa_utils.update_bn(train_loader, self.swa_model, device=self.device)

            # what are the weights and grads of the swa_model at the end of training?
            # weights, grads = get_weights_and_grads(self.model)
            # print(f'model params before swa return:\nP_i vec: {weights[:10]}\nD_i vec: {grads[:10]}\n')
            # swa_params, swa_derives = get_weights_and_grads(self.swa_model)
            # print(f'swa_model params before swa return:\nP_swa vec: {swa_params[:10]}\nD_swa vec: {swa_derives[:10]}\n')

            # save entire swa_model grads dict to csv file and return
            if self.args.record:
                df = pd.DataFrame(swa_epoch_derivatives)
                csv_file_path = os.path.join(path_D_i_swa, 'entire_swa_grads.csv')
                df.to_csv(csv_file_path, index=False)
                print(df.head(), df.shape, df.columns)


            return

        else:
            # swa_model was not used.
            best_model_path = os.path.join('./checkpoints/' + setting, 'best_epoch.pth')
            print(f'Loading from best_model_path:', {best_model_path})
            self.model.load_state_dict(torch.load(best_model_path), strict=False)

            # weights, grads = get_weights_and_grads(self.model)
            # print(f'model weights before regular return:\nweights vec: {weights[:10]}\ngradients vec: {grads[:10]}\n')

            if self.args.record:
                df = pd.DataFrame(epoch_grads)
                csv_file_path = os.path.join(path_G_i_original, 'entire_original_grads.csv')
                df.to_csv(csv_file_path, index=False)
                print(df.head(), df.shape, df.columns)


            return self.model


    def run_model(self, batch_x, batch_x_mark, batch_y, batch_y_mark): # METHOD NOT AMENDED!
        if self.args.model in ['NBEATS-G', 'NBEATS-I']:
            batch_x = batch_x.squeeze(-1)
            batch_y = batch_y[:, -self.args.pred_len:].squeeze(-1)
            input_mask = torch.ones(batch_x.shape).to(self.device)
            outputs = self.model(batch_x, input_mask)
        else:
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        return batch_y, outputs


    def test(self, setting, swa_flag=False):
        test_data, test_loader = self._get_data(flag='test')
        print('loading model')
        if swa_flag:
            test_model = self.swa_model
        else:
            test_model = self.model
            test_model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, f'best_epoch.pth')))
            # path = os.path.join(self.args.checkpoints, setting)
            # if self.args.delete_checkpoints:
            #     # delete the checkpoints
            #     try:
            #         shutil.rmtree(path)
            #     except OSError as e:
            #         print("Error: %s - %s." % (e.filename, e.strerror))
        w, g = get_weights_and_grads(test_model)
        print(f'model at start of test:\nweights vec: {w[:10]}\ngradients vec: {g[:10]}\n')

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
             os.makedirs(folder_path)

        test_model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model in ['PatchTST', 'DLinear']:
                            outputs = test_model(batch_x)
                        else:
                            my_tuple = MyTupleWrapper(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            if self.args.output_attention:
                                outputs = test_model(my_tuple)[0]
                            else:
                                outputs = test_model(my_tuple)
                else:
                    if self.args.model in ['PatchTST', 'DLinear']:
                        outputs = test_model(batch_x)
                    else:
                        my_tuple = MyTupleWrapper(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        if self.args.output_attention:
                            outputs = test_model(my_tuple)[0]

                        else:
                            outputs = test_model(my_tuple)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()

        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        print(f'\ntest shape:', preds.shape, trues.shape, inputx.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])
        print('test shape:', preds.shape, trues.shape, inputx.shape)

        # result save
        # folder_path = './overleaf/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)

        data = 'ILI' if self.args.data_path.split('.')[0] == 'national_illness' else self.args.data_path.split('.')[0]
        if self.args.use_neptune:
            id = self.args.neptune_run["sys/id"].fetch().split("-", 2)[1]
            if self.args.train_epochs < self.args.swa_start:
                file = f"results_original.csv"
                fields = ['Neptune', 'itr', 'Model', 'Dataset', 'Horizon', 'Type', 'MSE', 'MAE']
                row = {'Neptune': id, 'itr': str(self.args.seed)+'_'+str(self.args.ii), 'Model': self.args.model, 'Dataset': data,
                       'Horizon': self.args.pred_len, 'Type': self.args.features, 'MSE': mse, 'MAE': mae}

            else:
                file = f"results_swa.csv"
                fields = ['Neptune_swa', 'itr', 'Model', 'Dataset', 'Horizon', 'Type', 'SWA_start', 'anneal_epochs', 'learning_rate', 'MSE_swa', 'MAE_swa']
                row = {'Neptune_swa': id, 'itr': str(self.args.seed)+'_'+str(self.args.ii), 'Model': self.args.model, 'Dataset': data,
                       'Horizon': self.args.pred_len, 'Type': self.args.features, 'SWA_start': self.args.swa_start,
                       'anneal_epochs': self.args.anneal_epochs, 'learning_rate': self.args.swa_lr, 'MSE_swa': mse, 'MAE_swa': mae}

            file = "/cs_storage/yuvalao/code/latex/" + file
            with open(file, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fields)
                if not os.path.isfile(file):
                    writer.writeheader()
                writer.writerow(row)
                csvfile.close()

        print(f'model: {self.args.model}, data: {data}, horizon: {self.args.pred_len}, type: {self.args.features}, mse: {mse}, mae: {mae}, mspe: {mspe}, mape: {mape}')

        # f = open("result_long_term_forecast.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        # f.write('\n')
        # f.write('\n')
        # f.close()
        #
        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        if self.args.use_neptune:
            self.args.neptune_run["eval/mae"+'_swa'*swa_flag] = mae
            self.args.neptune_run["eval/mse"+'_swa'*swa_flag] = mse
            self.args.neptune_run["eval/rmse"+'_swa'*swa_flag] = rmse
            self.args.neptune_run["eval/mape"+'_swa'*swa_flag] = mape
            self.args.neptune_run["eval/mspe"+'_swa'*swa_flag] = mspe


        # if not self.args.record:
        #     shutil.rmtree(os.path.join('./checkpoints/', setting))
        # else:
        #     checkpoint_dir = './checkpoints/' + setting
        #     best_epoch_path = os.path.join(checkpoint_dir, f'best_epoch.pth')
        #     os.rename(best_epoch_path, os.path.join(checkpoint_dir, f'{best_epoch}.pth'))

        print(f'\nDone testing.')
        return


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.model in ['NBEATS-G', 'NBEATS-I']:
                    batch_x = batch_x.squeeze(-1)
                    batch_y = batch_y[:, -self.args.pred_len:].squeeze(-1)
                    input_mask = torch.ones(batch_x.shape).to(self.device)
                    outputs = self.model(batch_x, input_mask)
                else:
                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.model in ['PatchTST', 'DLinear']:
                                outputs = self.model(batch_x)
                            else:
                                my_tuple = MyTupleWrapper(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                if self.args.output_attention:
                                    outputs = self.model(my_tuple)[0]
                                else:
                                    outputs = self.model(my_tuple)
                    else:
                        if self.args.model in ['PatchTST', 'DLinear']:
                            outputs = self.model(batch_x)
                        else:
                            my_tuple = MyTupleWrapper(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            if self.args.output_attention:
                                outputs = self.model(my_tuple)[0]
                            else:
                                outputs = self.model(my_tuple)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        # folder_path = './overleaf/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        #
        # np.save(folder_path + 'real_prediction.npy', preds)

        return
