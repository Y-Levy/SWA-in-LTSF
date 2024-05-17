import os
import glob
import math
import tqdm
import torch
import numpy as np
import random
import torch
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
import polars as pl
import copy

plt.switch_backend('agg')

def adjust_learning_rate_TST(optimizer, scheduler, epoch, args):
    if args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
    return lr

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
    return lr


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_epoch = 1

    def __call__(self, val_loss, model, path, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_epoch = epoch
            self.best_score = score
            print(f'saving model to {path}/checkpoint_ep{self.best_epoch}.pth ......')
            self.save_checkpoint(val_loss, model, path) #, epoch=self.best_epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            #TODO: take ESD picture
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_epoch = epoch
            self.best_score = score
            self.save_checkpoint(val_loss, model, path) #, epoch=self.best_epoch)
            self.counter = 0
        print(f'\t\t------best epoch: {self.best_epoch}')
        return self.best_epoch

    def save_checkpoint(self, val_loss, model, path): #, epoch):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #torch.save(model.state_dict(), path + '/' + 'checkpoint.pth') #f'{path}/checkpoint_ep{epoch}.pth')
        #print(glob.glob('*.pth'))
        #print(f'saving model to {path}/checkpoint_ep{epoch}.pth ...')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def read_grads_file(path, filename):
    csv_file_path = os.path.join(path, filename)
    df = pd.read_csv(csv_file_path)
    print(f'{round(df.iloc[0][0], 8)}\n{round(df.iloc[0][1], 8)}')
    grads_dict = df.to_dict(orient='list')
    epochs_list = [int(f.split('_')[1]) for f in df.columns]

    return grads_dict, epochs_list


def get_weights_and_grads(model):
    """Current weights and gradients of the model, as a list."""
    weights = []
    gradients = []
    for name, parm in model.named_parameters():
        weights.extend(parm.detach().flatten().cpu().numpy())
        if parm.grad is not None:
            gradients.extend(parm.grad.flatten().cpu().numpy())

    return weights, gradients


def compute_norms(vec: list):
    grads_l1_norm = torch.norm(torch.tensor(vec), 1)
    grads_l2_norm = torch.norm(torch.tensor(vec), 2)

    if vec is not None:
        grads_max_norm = torch.norm(torch.tensor(vec), float('inf'))
        print(f'grads_l1_norm: {round(grads_l1_norm.item(), 4)}\ngrads_l2_norm: {round(grads_l2_norm.item(), 4)}\ngrads_max_norm: {round(grads_max_norm.item(), 4)}')
        return grads_l1_norm.item(), grads_l2_norm.item(), grads_max_norm.item()
    else:
        print(f'grads_l1_norm: {round(grads_l1_norm.item(), 4)}\ngrads_l2_norm: {round(grads_l2_norm.item(), 4)}')
        return grads_l1_norm.item(), grads_l2_norm.item()


def get_epochs_to_aggregate(best_epoch, start_epoch, ensemble_size):
    """Given the best epoch and the percentage of epochs to aggregate, return the list of the epochs to aggregate."""
    if ensemble_size is not None and ensemble_size > 0 and ensemble_size < 1 and start_epoch is None:
        num_epochs_to_average = 1 if (ensemble_size * best_epoch < 1) else math.floor(best_epoch * ensemble_size)
        last_epochs_range = list(range(best_epoch - num_epochs_to_average + 1, best_epoch + 1))
    elif ensemble_size is not None and ensemble_size >= 1 and start_epoch is None:
        start_epoch = best_epoch - int(ensemble_size) + 1
        last_epochs_range = [epoch for epoch in range(start_epoch, best_epoch + 1)]
    elif start_epoch is not None and ensemble_size is None:
        last_epochs_range = [epoch for epoch in range(start_epoch, best_epoch + 1)]
    else:
        raise ValueError("Either 'swa_percent' or 'start_epoch' must be provided.")

    return last_epochs_range


def get_weights_dict(model, path, map_location='cuda:0'):
    """
    Given a path to a folder containing the weights of the model in each epoch, return a dictionary of the weights in each epoch.
    :param path: path to folder containing the weights of the model in each epoch
    :param map_location:
    :return:
    """
    epochs_list = [int(f.split('_')[1].split('.pth')[0]) for f in os.listdir(path)]
    epochs_list.sort()
    epochs_start, epochs_finish = min(epochs_list), max(epochs_list)
    weights_dict = {}
    for i in range(epochs_start, epochs_finish + 1):
        model_in_epoch = model
        model_in_epoch.load_state_dict(torch.load(os.path.join(path, f'epoch_{i}.pth'), map_location=map_location))
        weights, grads = get_weights_and_grads(model_in_epoch)
        weights_dict[f'epoch_{i}'] = weights

    return weights_dict, epochs_list


def compute_distance(w_1, w_2):
    w_1 = torch.tensor(w_1)
    w_2 = torch.tensor(w_2)
    assert len(w_1) == len(w_2), "Weights must have the same number of layers"

    l1_norm = torch.norm(w_1 - w_2, p=1) / torch.norm(w_1, p=1)
    l2_norm = torch.norm(w_1 - w_2, p=2) / torch.norm(w_1, p=2)
    max_norm = torch.norm(w_1 - w_2, p=float('inf')) / torch.norm(w_1, p=float('inf'))
    cosine_similarity = F.cosine_similarity(w_1, w_2,0 )
    print(f'i.e: {[round(w.item(), 8) for w in w_1[:10]]} to\n\t {[round(w.item(), 8) for w in w_2[:10]]}\n\nl1_norm: {l1_norm}\nl2_norm: {l2_norm}\nmax_norm: {max_norm}\ncosine_similarity: {cosine_similarity}\n')

    return l1_norm, l2_norm, max_norm, cosine_similarity


def heatmap_comparisons(dict_1, dict_2, epochs_list_1, epochs_list_2):
    if len(epochs_list_1) < len(epochs_list_2):
        final_epochs_list = epochs_list_1
        sliced_dict_2 = dict_2
        sliced_dict_1 = {}
        for epoch, weights in dict_1.items():
            if int(epoch.split('_')[1]) in epochs_list_2:
                sliced_dict_1[epoch] = weights
    elif len(epochs_list_1) > len(epochs_list_2):
        final_epochs_list = epochs_list_2
        sliced_dict_1 = dict_1
        sliced_dict_2 = {}
        for epoch, weights in dict_2.items():
            if int(epoch.split('_')[1]) in epochs_list_1:
                sliced_dict_2[epoch] = weights
    else:
        final_epochs_list = epochs_list_1
        sliced_dict_1 = dict_1
        sliced_dict_2 = dict_2

    dim = min(len(epochs_list_1), len(epochs_list_2))
    start = max(epochs_list_1[0], epochs_list_2[0])
    l1_norm_heatmap = np.zeros(shape=(dim, dim))
    l2_norm_heatmap = np.zeros(shape=(dim, dim))
    max_norm_heatmap = np.zeros(shape=(dim, dim))
    cos_sim_heatmap = np.zeros(shape=(dim, dim))

    for i in range(dim):
        for j in range(dim):
            if i + start == 76 or j + start == 76:
                continue
            print(f'Comparing: epoch_{i + start} to epoch_{j + start}')
            l1_norm, l2_norm, max_norm, cos_sim = compute_distance(sliced_dict_1[f'epoch_{i + start}'], sliced_dict_2[f'epoch_{j + start}'])
            l1_norm_heatmap[i][j] = l1_norm
            l2_norm_heatmap[i][j] = l2_norm
            max_norm_heatmap[i][j] = max_norm
            cos_sim_heatmap[i][j] = cos_sim

    l1_norm_heatmap = np.round(l1_norm_heatmap[::-1], 4)
    l2_norm_heatmap = np.round(l2_norm_heatmap[::-1], 4)
    max_norm_heatmap = np.round(max_norm_heatmap[::-1], 4)
    cos_sim_heatmap = np.round(cos_sim_heatmap[::-1], 4)

    return l1_norm_heatmap, l2_norm_heatmap, max_norm_heatmap, cos_sim_heatmap, final_epochs_list


def plot_heatmaps(dict_1, dict_2, epochs_list, l1_norm_heatmap, l2_norm_heatmap, max_norm_heatmap, cos_sim_heatmap, title):
    finish = max(epochs_list)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    if 'Weight' in title:
        fig.suptitle(f'{title}')
    else:
        if 'base-model' in title.lower():
            fig.suptitle('Gradients: Magnitudes of the difference between base-model (G_i) and SWA-model (G_j), relative to the respective norm of (G_i)')
        else:
            fig.suptitle('Gradients: Magnitudes of the difference between SWA-model (G_j) and base-model (G_i), relative to the respective norm of (G_j)')
    similarity_measures = [l1_norm_heatmap, l2_norm_heatmap, max_norm_heatmap, cos_sim_heatmap]
    labels = ['l1_norm', 'l2_norm', 'max_norm', 'cosine_similarity']
    epochs_list.sort()

    for i, ax in enumerate(axes.flat):
        if labels[i] == 'cosine_similarity':
            vmin, vmax = -1, 1
        else:
            vmin, vmax = 0, np.max(similarity_measures[i])
        sns.heatmap(similarity_measures[i], cmap='RdBu_r', ax=ax, xticklabels=epochs_list, yticklabels=epochs_list[::-1], vmin=vmin, vmax=vmax)
        ax.set_title(labels[i])
        if 'base-model' in title.lower():
            if 'Weight' in title:
                ax.set_xlabel('base-model (W_i) Epochs')
                ax.set_ylabel('SWA-model (W_j) Epochs')
            else:
                ax.set_xlabel('SWA-model (G_j) Epochs')
                ax.set_ylabel('base-model (G_i) Epochs')
        else:
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Epochs')

        if finish < 100: #to mmake sure this doesn't apply to PathTST
            num_tiles = len(epochs_list)
            font_size = 12 if num_tiles <= 5 else 12 * (5 / num_tiles)
            for j in range(len(epochs_list)):
                for k in range(len(epochs_list)):
                    if labels[i] == 'cosine_similarity':
                        ax.text(k + 0.5, j + 0.5, f'{similarity_measures[i][j, k]:.2f}', ha='center', va='center', fontsize=font_size, color='lime')
                    else:
                        ax.text(k + 0.5, j + 0.5, f'{similarity_measures[i][j, k]:.5f}', ha='center', va='center', fontsize=font_size, color='lime')

    plt.tight_layout()
    plt.show()

    return fig


def weight_comparison_exp(dict_1, dict_2, epochs_list_1, epochs_list_2, header):
    print(f'{header}:')
    print(f'producing heatmaps...\n')
    l1_norm_heatmap, l2_norm_heatmap, max_norm_heatmap, cos_sim_heatmap, epochs = heatmap_comparisons(dict_1, dict_2, epochs_list_1, epochs_list_2)
    output_plot = plot_heatmaps(dict_1, dict_2, epochs_list_2, l1_norm_heatmap, l2_norm_heatmap, max_norm_heatmap, cos_sim_heatmap, title=header)

    return output_plot


def random_color():
    # Generate random RGB values
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)

    # Convert to a valid color format (e.g., hexadecimal)
    color = "#{:02x}{:02x}{:02x}".format(r, g, b)

    return color


class MyTupleWrapper:
    def __init__(self, val1, val2, val3, val4):
        self.val1, self.val2, self.val3, self.val4, self.device  = val1, val2, val3, val4, 'cuda'

    def cuda(self):
        self.val1 = self.val1.cuda()
        self.val2 = self.val2.cuda()
        self.val3 = self.val3.cuda()
        self.val4 = self.val4.cuda()
        return self

    def __repr__(self):
        return f'MyTupleWrapper({self.val1, self.val2, self.val3, self.val4, self.device})'


def ensemble_P_i(params_path, best_epoch, start_epoch, ensemble_size, custom_averaging, eps=None):
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
        eps * averaged_model_parameter + (1-eps) * model_parameter

    for i in swa_epochs:
        model_params_dict[f'epoch_{i}'] = torch.load(os.path.join(params_path, f'epoch_{i}.pth'), map_location='cuda:0')
        print(f'epoch_{i} loaded: {round(model_params_dict[f"epoch_{i}"]["enc_embedding.value_embedding.tokenConv.weight"][:1][0][0][0].item(), 10)}')


    random_epoch = random.choice(swa_epochs)
    for layer in model_params_dict[f'epoch_{random_epoch}'].keys():
        layer_params_sum = None

        for epoch in swa_epochs:
            epoch_params = model_params_dict[f'epoch_{epoch}'][layer]

            if layer_params_sum is None:
                layer_params_sum = epoch_params.clone()
            else:
                if custom_averaging:
                    layer_params_sum = ema_avg(layer_params_sum, epoch_params, epoch - min(swa_epochs) + 1)
                else:
                    layer_params_sum += epoch_params

        mean_params = layer_params_sum / len(swa_epochs)
        swa_param_dict[layer] = mean_params

    avg = round(swa_param_dict['enc_embedding.value_embedding.tokenConv.weight'][:1][0][0][0].item(), 10)
    print(f'SWA_avg: {avg}')
    torch.save(swa_param_dict, params_path + 'avg_temp.pth')
    return