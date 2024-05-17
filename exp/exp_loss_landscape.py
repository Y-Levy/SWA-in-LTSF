__all__ = ['Exp_LossLandscape']
import os
import sys
import math
import torch
import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
from utils.metrics import metric
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import (get_epochs_to_aggregate, get_weights_dict, weight_comparison_exp, get_weights_and_grads,
                         compute_norms, read_grads_file, random_color, MyTupleWrapper)

class Exp_LossLandscape(Exp_Basic):
    def __init__(self, args):
        super(Exp_LossLandscape, self).__init__(args)
        self.swa_start = args.swa_start

    def _build_model(self, args):
        if self.args.model == 'NBEATS-G':
            model = self.model_dict[self.args.model](input_size=self.args.seq_len,
                                                output_size=self.args.pred_len,
                                                stacks=self.args.nb_stacks,
                                                layers=self.args.nb_layers,
                                                layer_size=self.args.nb_layer_size).float()
        elif self.args.model == 'NBEATS-I':
            model = self.model_dict[self.args.model](input_size=self.args.seq_len,
                                                output_size=self.args.pred_len,
                                                trend_blocks=self.args.nb_trend_blocks,
                                                trend_layers=self.args.nb_trend_layers,
                                                trend_layer_size=self.args.nb_trend_layer_size,
                                                seasonality_blocks=self.args.nb_seasonality_blocks,
                                                seasonality_layers=self.args.nb_seasonality_layers,
                                                seasonality_layer_size=self.args.nb_seasonality_layer_size,
                                                num_of_harmonics=self.args.nb_num_of_harmonics,
                                                degree_of_polynomial=self.args.nb_degree_of_polynomial).float()
        else:
            model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader


    def ensemble_P_i(self, params_path, best_epoch, start_epoch, ensemble_size):
        """Given the location of P_i from different epochs, ensemble size -
               calculate SWA weights (P_swa) and save them in a temp.pth file

           Optional arguments:
               best_epoch - the epoch with the best performance (default: self.args.train_epochs)
                start_epoch - the epoch to start the SWA from (default: self.args.swa_start)
        """
        # if best_epoch is None:
        #     best_epoch = self.args.train_epochs
        # if start_epoch is None:
        #     start_epoch = self.args.swa_start
        swa_epochs = get_epochs_to_aggregate(best_epoch, start_epoch, ensemble_size)

        model_params_dict = {}
        swa_param_dict = {}
        ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: \
            0.1 * averaged_model_parameter + 0.9 * model_parameter

        for i in swa_epochs:
            model_params_dict[f'epoch_{i}'] = torch.load(os.path.join(params_path, f'epoch_{i}.pth'), map_location='cuda:0')

        random_epoch = random.choice(swa_epochs)
        for layer in model_params_dict[f'epoch_{random_epoch}'].keys():
            layer_params_sum = None

            for epoch in swa_epochs:
                epoch_params = model_params_dict[f'epoch_{epoch}'][layer]

                if layer_params_sum is None:
                    layer_params_sum = epoch_params.clone()
                else:
                    if self.args.custom_averaging:
                        layer_params_sum = ema_avg(layer_params_sum, epoch_params, epoch - min(swa_epochs) + 1)
                    else:
                        layer_params_sum += epoch_params

            mean_params = layer_params_sum / len(swa_epochs)
            swa_param_dict[layer] = mean_params

        torch.save(swa_param_dict, params_path + 'temp.pth')
        return


    def interpolation_preformance(self, setting, start_epoch, lambda_scaler):
        """Given the start epoch, lambda value -
               calculate performance metrics of the interpolation with the last epoch (fixed, given through 'setting')

           Returns
               mae, mse, rmse, mape, mspe - float values of the results
        """
        path_original, path_swa, map_location = os.path.join(self.args.checkpoints, setting + '/W_i_original/'), os.path.join(self.args.checkpoints, setting + '/P_i_swa/'), 'cuda:0'
        end_epoch = self.args.train_epochs

        self.ensemble_P_i(path_swa, best_epoch=end_epoch, start_epoch=start_epoch, ensemble_size=None)
        start_epoch_dict = torch.load(os.path.join(path_swa, 'temp.pth'))
        end_epoch_dict = torch.load(os.path.join(path_original, f'epoch_{end_epoch}.pth'), map_location=map_location)

        os.remove(path_swa + 'temp.pth')

        interpolated_weights = {}
        for layer in start_epoch_dict.keys():
            start = start_epoch_dict[layer].float()
            end = end_epoch_dict[layer].float()
            interpolated_weights[layer] = torch.lerp(start, end, torch.full_like(start, lambda_scaler))

        torch.save(interpolated_weights, path_swa + f'interpolation.pth')
        preds, trues = self.test(path_swa + f'interpolation.pth')
        os.remove(path_swa + 'interpolation.pth')

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f'\u03BB: {lambda_scaler}, mae: {mae}, mse: {mse}, rmse: {rmse}, mape: {mape}, mspe: {mspe}')
        return mae, mse, rmse, mape, mspe


    def get_swa_params_dict(self, setting):
        """Returns a dictionary of the params of the SWA model in each epoch, and a list of the epochs"""
        lr, ae, t = setting.split('_')[4], setting.split('_')[5], setting.split('_')[3]
        path, map_location = os.path.join(self.args.checkpoints, setting + f'/P_i_swa_{lr}_{ae}_{t}/'), 'cuda:0'
        epochs_list = [int(f.split('_')[1].split('.pth')[0]) for f in os.listdir(path)]
        epochs_list.sort()
        epochs_start, epochs_finish = min(epochs_list), max(epochs_list)
        params_dict = {}
        for i in range(epochs_start, epochs_finish + 1):
            self.ensemble_P_i(path, best_epoch=i, start_epoch=epochs_start, ensemble_size=None)
            model_in_epoch = torch.load(os.path.join(path, f'temp.pth'), map_location=map_location)
            os.remove(path + 'temp.pth')
            params = []
            for layer in model_in_epoch.keys():
                layer_params = model_in_epoch[layer]
                params.extend(layer_params.flatten().cpu().numpy())

            params_dict[f'epoch_{i}'] = params

        print(f'\nswa_epochs_list: {epochs_list}\nswa_params_dict: ')
        for key, value in params_dict.items():
            print(f'{key}: {value[:10]}\n')

        return params_dict, epochs_list


    def test(self, path):
        self.model.load_state_dict(torch.load(path))
        test_data, test_loader = self._get_data(flag='test')
        preds = []
        trues = []
        inputx = []
        self.model.eval()
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

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)

        print(f'\ntest shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        return preds, trues


    def W_swa_preds(self, setting, ensemble_size):
        """
        Returns prediction of model with loaded SWA weights
        """
        lr, ae, t = setting.split('_')[4], setting.split('_')[5], setting.split('_')[3]
        path_P_i_swa = os.path.join(self.args.checkpoints, setting + f'/P_i_swa_{lr}_{ae}_{t}/')
        # if self.args.train_epochs >= 10:
        #     ensemble_size /= self.args.train_epochs
        # else:
        #     ensemble_size /= 10
        self.ensemble_P_i(path_P_i_swa, best_epoch=self.args.train_epochs, start_epoch=None, ensemble_size=ensemble_size)
        preds, trues = self.test(path_P_i_swa + 'temp.pth')
        os.remove(path_P_i_swa + 'temp.pth')

        return preds, trues


    def plot_data(self, setting):
        """loop through all ensemble sizes and calculate for each 4 performance metrics

           Returns
               Plot data -
               A dict with 4 keys (mae, mse, mape, mspe) and values for the different ensemble sizes

               MAE, MSE line values -
               mae, mse (baseline) results for the given setting (with early stopping)
        """
        plot_data = {str(metric): [] for metric in ["mae", "mse", "mape", "mspe"]}
#        preds_original, trues_original = self.test(os.path.join(self.args.checkpoints, setting + '/checkpoint.pth'))
#        mae_line_value, mse_line_value = metric(preds_original, trues_original)[0], metric(preds_original, trues_original)[1]
        if setting.split("_")[2] == 'PatchTST':
            mae_line_value, mse_line_value = 0.3282, 0.2707
        elif setting.split("_")[2] == 'Autoformer':
            mae_line_value, mse_line_value = 0.3381, 0.2723
        else:
            mae_line_value, mse_line_value = 0.0869, 0.011

        max_ensemble_size = self.args.train_epochs-self.args.swa_start+1
        final_swa_start = 80 if self.args.model == 'PatchTST' else 8
        min_ensemble_size = self.args.train_epochs-final_swa_start
        step = self.args.train_epochs/-10 if self.args.train_epochs >= 10 else -1
        ensemble_sizes = [idx for idx in range(max_ensemble_size, min_ensemble_size, int(step))]

        for ensemble_size in tqdm(ensemble_sizes):
            preds, trues = self.W_swa_preds(setting, ensemble_size)
            mae, mse, rmse, mape, mspe = metric(preds, trues)
            for metric_name, metric_value in zip(["mae", "mse", "mape", "mspe"], [mae, mse, mape, mspe]):
                plot_data[metric_name].append(metric_value)

        return plot_data, mae_line_value, mse_line_value


    def interpolation_plot_data(self, setting, start_epoch):
        """Given the start epoch, loop through 10 lambda values and calculate for each 4 performance metrics

           Returns
               Plot data -
               A dict with 4 keys (mae, mse, mape, mspe) and 10 values for each key, corresponding to the
               interpolation results of the start epoch, last epoch (fixed and given through 'setting'), and lambda.
        """
        plot_data = {str(metric): [] for metric in ["mae", "mse", "mape", "mspe"]}
        lambdas = torch.linspace(0, 1, 10)
        for lambda_scaler in tqdm(lambdas):
            mae, mse, rmse, mape, mspe = self.interpolation_preformance(setting, start_epoch, lambda_scaler)
            print(f'\u03BB: {lambda_scaler}, mae: {mae}, mse: {mse}, rmse: {rmse}, mape: {mape}, mspe: {mspe}')
            for metric_name, metric_value in zip(["mae", "mse", "mape", "mspe"], [mae, mse, mape, mspe]):
                plot_data[metric_name].append(metric_value)

        return plot_data


    def interpolation_exp(self, setting):
        path, map_location = os.path.join(self.args.checkpoints, setting + '/P_i_swa/'), 'cuda:0'
        #To EXCLUDE the last epoch star: # if int(f.split('_')[1].split('.pth')[0]) < self.args.train_epochs]
        if self.args.model == 'PatchTST':
            epochs_list = [int(f.split('_')[1].split('.pth')[0]) for f in os.listdir(path) if int(f.split('_')[1].split('.pth')[0])%10==0]
        else:
            epochs_list = [int(f.split('_')[1].split('.pth')[0]) for f in os.listdir(path)]
        epochs_list.sort()
        all_interpolations = {}
        color_by_epoch = {}
        metric_limits = {metric: (float('inf'), float('-inf')) for metric in ["mae", "mse", "mape", "mspe"]}

        for epoch in tqdm(epochs_list):
            print(f'\nfrom epoch: {epoch}...')
            color_by_epoch[epoch] = random_color()
            all_interpolations[f'from epoch_{epoch}'] = self.interpolation_plot_data(setting, epoch)
            for metric in metric_limits:
                metric_limits[metric] = (min(metric_limits[metric][0],
                                             min(all_interpolations[f'from epoch_{epoch}'][metric])),
                                         max(metric_limits[metric][1],
                                             max(all_interpolations[f'from epoch_{epoch}'][metric])))

        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        x_labels = [idx.item() for idx in torch.linspace(0,1,5)]
        metrics = ["mae", "mse", "mape", "mspe"]
        data = 'ILI' if self.args.data_path.split('.')[0] == 'national_illness' else self.args.data_path.split('.')[0]
        fig.suptitle(f'model: {self.args.model}, data: {data}, horizon: {self.args.pred_len}, type: {self.args.features}')

        for i in range(2):
            for j in range(2):
                metric = metrics[i * 2 + j]
                ax = axs[i, j]
                ax.set_xlabel('\u03BB')
                ax.set_ylabel(f'Test {metric}')

                for epoch in all_interpolations.keys():
                    values = all_interpolations[epoch][metric]
                    ax.plot(torch.linspace(0, 1, 10), values, label=epoch, color=color_by_epoch[int(epoch.split('_')[1])])

                ax.grid(True)
                ax.set_xlim(0, 1)
                ax.set_xticks(x_labels)
                ax.set_xticklabels(x_labels)
                ax.set_ylim(metric_limits[metric][0], metric_limits[metric][1])
                if i==0 and j==1:
                    #ncols = 7 if self.args.model == 'PatchTST' else 1
                    ax.legend(loc='upper right', bbox_to_anchor=(1.5, 1.15), ncol=1)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.grid(True)
        plt.show()

        return fig


    def ensemble_size_comparison_exp(self, setting):
        data_dict, mae_line_value, mse_line_value = self.plot_data(setting)
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))

        max_ensemble_size = self.args.train_epochs-self.args.swa_start+1
        final_swa_start = 80 if self.args.model == 'PatchTST' else 8
        min_ensemble_size = self.args.train_epochs-final_swa_start
        step = self.args.train_epochs/-10 if self.args.train_epochs >= 10 else -1
        ensemble_sizes = list(range(max_ensemble_size, min_ensemble_size, int(step)))

        metrics = ["mae", "mse", "mape", "mspe"]
        data = 'ILI' if self.args.data_path.split('.')[0] == 'national_illness' else self.args.data_path.split('.')[0]
        fig.suptitle(f'swa_lr: {self.args.swa_lr}, anneal_epochs: {self.args.anneal_epochs}') #(f'Model: {self.args.model}, Data: {data}, Horizon: {self.args.pred_len}, Type: {self.args.features}\n\nfinal epoch: {self.args.train_epochs}, swa start: {self.args.swa_start}.')

        for i in range(2):
            for j in range(2):
                metric = metrics[i * 2 + j]

                values = data_dict[metric]
                sorted_indices = np.argsort(values)
                values = [values[i] for i in sorted_indices]
                x = [str(ensemble_sizes[i]) for i in sorted_indices]

                colors = plt.cm.Blues(np.linspace(0.2, 0.8, len(x)))
                df = pd.DataFrame({'Size': x, 'Result': values})

                ax = axs[i, j]
                ax.set_xlabel('Ensemble Sizes')
                ax.set_ylabel(f'Test {metric}')
                bars = ax.bar(x, values, color=colors, label=f'{self.args.model}', width=.35)

                if metric == 'mae':
                    line_value = mae_line_value
                elif metric == 'mse':
                    line_value = mse_line_value
                else:
                    line_value = None

                # add a horizontal line for the baseline
                # if line_value is not None:
                #     ax.axhline(y=line_value, color='red', linestyle='--', linewidth=2, label='Calculated baseline (with early stopping)')
                #     ax.text(0.5, line_value, str(round(line_value, 3)), va='center', ha='center', color='black', fontsize = 10,
                #             backgroundcolor='w', transform=ax.get_yaxis_transform())
                #
                # add the values for each bar
                for bar, value in zip(bars, values):
                    if line_value is None:
                        ax.text(bar.get_x() + bar.get_width()/2, value/2, str(round(value, 3)), ha='center', color='black')
                    else:
                        if value <= line_value:
                            ax.text(bar.get_x() + bar.get_width()/2, value/2, str(round(value, 3)), ha='center', color='black') #'green'
                        else:
                            ax.text(bar.get_x() + bar.get_width()/2, value/2, str(round(value, 3)), ha='center', color='black') #'red'


                ax.set_title(f'Metric: {metric}')
                ax.set_xticks(x)
                fig.legend([f'Calculated baseline.\n(with early stopping)'], loc='upper right')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        return fig

        #TODO: plot forecasted series results. Note DTW Barycenter Averaging is usually for series of different lengths - parse 720 to 96??
        #TODO: swa_patience - does averaging start at the early stop? if not, when do we start averaging?
        #TODO: random init of weights
        #TODO: check diversity of the averaged SWA weights - W_j
        #TODO: check accuracy of the averaged SWA weights W_j - f(W_j)
        #TODO: similarity measure of solutions from SWA weights W_j - how?
        #TODO: Domain specific effects on Long-Term Forecasting: seasonality? trend? noise? sparsity? singular spectrum analysis (SSA)?


    def interpolation_comparison_exp(self, setting, neptune_run):
        # Note: to make a valid comparison, W_i_original is WITHOUT earlu stopping
        figure = self.interpolation_exp(setting)
        neptune_run[f'loss_landscape/Intrpolation: P_swa vs. W_{self.args.train_epochs}(original)'].upload(figure)


    def ensemble_size_comparison(self, setting, neptune_run):
        figure = self.ensemble_size_comparison_exp(setting)
        neptune_run[f'loss_landscape/Ensemble Sizes'].upload(figure)


    def gradients_comparison(self, setting, neptune_run):
        path_G_i_original = os.path.join(self.args.checkpoints, setting + '/Grad_i_original/')
        path_D_i_swa = os.path.join(self.args.checkpoints, setting + '/D_i_swa/')

        print('\nGradients comparison. Loading gradients from swa run...\n')
        swa_epoch_grads, swa_epochs_list = read_grads_file(path_D_i_swa, 'entire_swa_grads.csv')
        exp_types = ['Gradients norms: Model during SWA to itself',
                     'Gradients norms: Base-model to model during SWA']

        for exp_type in exp_types:
            print(f'{exp_type}')
            if exp_type == exp_types[0]:
                figure = weight_comparison_exp(swa_epoch_grads, swa_epoch_grads, swa_epochs_list, swa_epochs_list,
                                               exp_type)

            elif os.path.exists(os.path.join(path_G_i_original, 'entire_original_grads.csv')) and exp_type == \
                    exp_types[1]:

                epoch_grads, base_epochs_list = read_grads_file(path_G_i_original, 'entire_original_grads.csv')
                figure = weight_comparison_exp(epoch_grads, swa_epoch_grads, base_epochs_list, swa_epochs_list,
                                               exp_type)

            neptune_run[f'loss_landscape/{exp_type}'].upload(figure)


    def weights_comparison(self, setting, neptune_run):
        print('\nWeights comparison. Loading weights from swa run...')
        path_W_i_original = os.path.join(self.args.checkpoints, setting + '/W_i_original/')
        path_P_i_swa = os.path.join(self.args.checkpoints, setting + '/P_i_swa/')
        P_i_weights_dict, P_i_epochs_list = get_weights_dict(model=self.model, path=path_P_i_swa)
        P_swa_weights_dict, P_swa_epochs_list = self.get_swa_weights_dict(setting)

        exp_types = ['Weights: Distances between different stages of SWA-model (P_i), relative to the respective norm of (P_i)',
                     'Weights: Distances between different stages of SWA-averages (P_swa), relative to the respective norm of (P_swa)',
                     'Weights: Distances between base-model (W_i) and SWA-model (P_i), relative to the respective norm of (W_i)']

        for exp_type in exp_types:
            title = 'Weights: '+exp_type.split('between ')[1].split(',')[0].capitalize()
            if exp_type == exp_types[0]:
                figure = weight_comparison_exp(P_i_weights_dict, P_i_weights_dict, P_i_epochs_list, P_i_epochs_list,
                                               exp_type)

            elif exp_type == exp_types[1]:
                figure = weight_comparison_exp(P_swa_weights_dict, P_swa_weights_dict, P_swa_epochs_list,
                                               P_swa_epochs_list, exp_type)

            elif exp_type == exp_types[2] and os.path.exists(path_W_i_original):
                W_i_weights_dict, W_i_epoch_list = get_weights_dict(model=self.model, path=path_W_i_original)
                figure = weight_comparison_exp(W_i_weights_dict, P_i_weights_dict, W_i_epoch_list, P_i_epochs_list,
                                               exp_type)

            neptune_run[f'loss_landscape/{title}'].upload(figure)


    def space_plot(self, setting, neptune_run):
        # given a swa exp. look for its original counterpart - return weights dict.
        path = ''
        for folder in os.listdir(self.args.checkpoints):
            if folder.endswith('test') and "_".join(setting.split("_")[:-5])==folder:
                path = folder
        path_W_i_original = os.path.join(self.args.checkpoints, path + '/W_i_original/')
        W_i_weights_dict, W_i_epochs_list = get_weights_dict(model=self.model, path=path_W_i_original)

        # given weights dict. - return df of (W_n - W_i)
        subs_df = {}
        last_epoch = "epoch_"+str(max(W_i_epochs_list))
        W_n = W_i_weights_dict[last_epoch]
        for key, value in W_i_weights_dict.items():
            if key == "epoch_"+str(max(W_i_epochs_list)):
                break
            else:
                temp = [x - y for x, y in zip(value, W_n)]
                subs_df[f'{key}-{last_epoch}'] = temp
        df = pd.DataFrame(subs_df)

        # PCA - torch
        scaler = StandardScaler()
        scaler.fit(df)
        df_scaled = scaler.transform(df)
        cov_matrix = torch.cov(torch.tensor(df_scaled).T)
        eig_values, eig_vectors = torch.linalg.eig(cov_matrix)
        top_eig_vectors = eig_vectors[:, :2].real

        # project data on the top 2 eigenvectors
        pca_result_torch = torch.matmul(torch.tensor(df_scaled), top_eig_vectors)
        pca1, pca2 = pca_result_torch.T[0], pca_result_torch.T[1]

        pca = PCA()
        principalComponents = pca.fit_transform(df_scaled)
        pc1, pc2 = pca.fit_transform(df_scaled)[:, 0], pca.fit_transform(df_scaled)[:, 1]
        print(torch.norm(torch.tensor(principalComponents[0]), p=2)-torch.norm(pca1, p=2))
        print(torch.norm(torch.tensor(principalComponents[1]), p=2) - torch.norm(pca2, p=2))
        print(F.cosine_similarity(pca1, torch.tensor(principalComponents[0]), 0))
        print(F.cosine_similarity(pca2, torch.tensor(principalComponents[1]), 0))

        # grid plot
        X, Y = torch.meshgrid(pca1, pca2)

        print(df.head(), df.shape, df.columns)