import argparse
import os
import torch
from exp.exp_loss_landscape import Exp_LossLandscape
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import random
import math
import numpy as np
import polars as pl
import pandas as pd
import neptune

def init_neptune(args):
    print("Initializing Neptune...")
    # create a neptune run object
    args.neptune_run = neptune.init_run(project=args.neptune_project, api_token=args.neptune_token)

    # Log parameters
    params = {
        'model_id': args.model_id,
        'model': args.model,
        'seq_len': args.seq_len,
        'label_len': args.label_len,
        'pred_len': args.pred_len,
        'features': args.features,
    }
    data = 'ILI' if args.data_path.split('.')[0] == 'national_illness' else args.data_path.split('.')[0]
    args.neptune_run['sys/tags'].add([args.model, data, str(args.pred_len), args.features])
    if args.swa_start <= args.train_epochs:
        params['swa'] = {
            'swa_lr': args.swa_lr,
            'swa_start': args.swa_start,
            'custom_averaging': args.custom_averaging,
            'anneal_epochs': args.anneal_epochs,
            'anneal_strategy': args.anneal_strategy,
        }
        args.neptune_run['sys/tags'].add(['swa', str(args.swa_start), str(args.swa_lr), str(args.anneal_epochs)])
    args.neptune_run["algorithm"] = args.model
    args.neptune_run["model/parameters"] = vars(args)
    args.neptune_run['parameters'] = params

def get_id(seed, ii, model, data_path, horizon, features, t, ae, lr):
    data = 'ILI' if data_path.split('.')[0] == 'national_illness' else data_path.split('.')[0]
    data_file = '/cs_storage/yuvalao/code/latex/results_swa.csv'
    seed = str(seed) + '_' + str(ii)
    setting = f'{seed}_{model}_{data}_{horizon}_{features}_{t}_{ae}_{lr}'
    results = []

    # Retrieve relevant ids
    with open(data_file, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            key = f"{parts[1]}_{parts[2]}_{parts[3]}_{parts[4]}_{parts[5]}_{parts[6]}_{parts[7]}_{parts[8]}"
            if setting == key:
                value = int(parts[0])
                #TODO: why not replace the max instead of appending?
                results.append(value)

    max_id = max(results)
    return f'SWA-{max_id}'

def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast', #required=True,
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=0, help='status') #required=True,
    parser.add_argument('--model_id', type=str, default='test', help='model id') #
    parser.add_argument('--model', type=str, default='Autoformer', #required=True,
                        help='model name, options: [Autoformer, FEDformer, PatchTST]')
    parser.add_argument('--seed', type=int, default=fix_seed, help='seed')
    parser.add_argument('--initialization_type', type=str, default='Kaiming He', help='weights initialization method, options: [base, Lecun, Xavier, Kaiming He]')

    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type') #required=True,
    parser.add_argument('--root_path', type=str, default='./dataset/illness/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='national_illness.csv', help='data file')
    parser.add_argument('--features', type=str, default='S',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--delete_checkpoints', action='store_false',
                        help='Delete the trained checkpoints after the model finished training', default=True)

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=36, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=18, help='start token length')
    parser.add_argument('--pred_len', type=int, default=36, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # supplementary config for FEDformer
    parser.add_argument('--version', type=str, default='Fourier', help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
    parser.add_argument('--mode_select', type=str, default='random', help='for FEDformer, there are two mode selection method, options: [random, low]')
    parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
    parser.add_argument('--L', type=int, default=3, help='ignore level')
    parser.add_argument('--base', type=str, default='legendre', help='mwt base')
    parser.add_argument('--cross_activation', type=str, default='tanh', help='mwt cross attention activation function tanh or softmax')

    # supplementary config for PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # supplementary config for N-BEATS-G
    parser.add_argument('--nb_layers', type=int, default=4, help='num of layers for nbeats only')
    parser.add_argument('--nb_layer_size', type=int, default=512, help='size of layers for nbeats only')
    parser.add_argument('--nb_stacks', type=int, default=30, help='num of stacks for nbeats only')

    # supplementary config for N-BEATS-I
    parser.add_argument('--nb_seasonality_layer_size', type=int, default=2048, help='num of layers for nbeats only')
    parser.add_argument('--nb_seasonality_blocks', type=int, default=3, help='size of layers for nbeats only')
    parser.add_argument('--nb_seasonality_layers', type=int, default=4, help='size of layers for nbeats only')
    parser.add_argument('--nb_trend_layer_size', type=int, default=256, help='size of layers for nbeats only')
    parser.add_argument('--nb_degree_of_polynomial', type=int, default=3, help='size of layers for nbeats only')
    parser.add_argument('--nb_trend_blocks', type=int, default=3, help='size of layers for nbeats only')
    parser.add_argument('--nb_trend_layers', type=int, default=4, help='size of layers for nbeats only')
    parser.add_argument('--nb_num_of_harmonics', type=int, default=1, help='size of layers for nbeats only')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=11, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--pct_start', type=float, default=0.4, help='pct_start')

    # swa
    parser.add_argument('--swa_lr', type=float, default=1e-4, help='swa scheduler')
    parser.add_argument('--swa_start', type=int, default=3, help='swa start epoch number')
    parser.add_argument('--swa_percent', type=float, default=0.5, help='percent of last epochs to be included in swa')
    parser.add_argument('--anneal_epochs', type=int, default=3, help='number of epochs in the annealing phase')
    parser.add_argument('--anneal_strategy', type=str, default='cos', help='annealing strategy to use, options: [cos, linear]')
    parser.add_argument('--custom_averaging', type=int, default=0, help='use a custom averaging strategy')

    # neptune
    parser.add_argument('--record', action='store_true', default=False, help='record epoch weights and grads for exp_loss_landscape')
    parser.add_argument('--use_neptune', action='store_true', default=False, help='use neptune, if set to True need to provide project and token, if False save locally')
    parser.add_argument('--neptune_project', type=str, default='azencot-group/SWA', help='neptune project name')
    parser.add_argument('--neptune_token', type=str,
                        default='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyZGE5Y2Y1ZC1iZTgxLTQwN2QtOTE1OC1lMGU0NWE0ZWRmMjQifQ==',
                        help='neptune api token')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128], help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):

            if args.use_neptune:
                init_neptune(args)
                args.neptune_run['sys/tags'].add([f'{ii}'])
            print(f'\n>>>>>>>>>>>>itr number: {ii}')

            # setting record of experiments
            data = args.data_path.split('.csv')[0]
            args.ii = ii

            setting = f'{args.seed}_{ii}_{args.model}_{data}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_fc{args.factor}_eb{args.embed}_dt{args.distil}_{args.des}'
            if args.swa_start <= args.train_epochs:
                setting += f'_s{args.swa_start}_lr{args.swa_lr}_ae{args.anneal_epochs}_ca{args.custom_averaging}_{args.anneal_strategy}'

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, swa_flag=True) if args.swa_start <= args.train_epochs else exp.test(setting)
            torch.cuda.empty_cache()
            if args.use_neptune:
                args.neptune_run.stop()
    else:
        ii = 0
        id = get_id(args.seed, ii, args.model, args.data_path, args.pred_len, args.features, args.swa_start, args.anneal_epochs, args.swa_lr)
        neptune_run = neptune.init_run(project=args.neptune_project, with_id=id, api_token=args.neptune_token)
        neptune_run['sys/tags'].add(['loss_landscape'])
        print(f'\n>>>>>>>>>>>>seed: {args.seed}__itr number: {ii}')

        data = args.data_path.split('.csv')[0]
        if args.swa_start > args.train_epochs:
            raise ValueError('loss landscape exp. are run with only SWA enabled!')
        setting = f'{args.seed}_{ii}_{args.model}_{data}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_fc{args.factor}_eb{args.embed}_dt{args.distil}_{args.des}'
        setting += f'_s{args.swa_start}_lr{args.swa_lr}_ae{args.anneal_epochs}_ca{args.custom_averaging}_{args.anneal_strategy}'

        args.ii = ii
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))

        #exp.interpolation_comparison_exp(setting, neptune_run)
        #exp.ensemble_size_comparison(setting, neptune_run)
        #exp.gradients_comparison(setting, neptune_run)
        #exp.weights_comparison(setting, neptune_run)
        exp.space_plot(setting, neptune_run)

        torch.cuda.empty_cache()
        neptune_run.stop()

if __name__ == '__main__':
    main()

