import argparse
import itertools
import os
from datetime import datetime

# use argparse to query for just_generate
parser = argparse.ArgumentParser()
parser.add_argument('--just_generate', action='store_true', default=False)
parser.add_argument('--just_run', action='store_true', default=False)
args = parser.parse_args()

models = ['FEDformer'] #'Autoformer', 'FEDformer', 'PatchTST'

# datasets_dims = {'ETT': 7, 'electricity': 321, 'exchange_rate': 8, 'traffic': 862, 'weather': 21, 'national_illness': 7}
# datasets_lengths = {ETTh: 17,420, ETTm: 69,680, electricity: 26,304, exchange_rate: 7,588, traffic: 17,544, weather: 52,696, national_illness: 966}
datasets = ['weather'] #['weather', 'traffic', 'ETTm2', 'electricity', 'exchange_rate', 'national_illness']

features = ['S'] #'S', 'M'
general_pred_lengths = [96] #96, 192, 336, 720
ili_pred_lengths = [36] #24, 36, 48, 60

just_generate = args.just_generate

append_file = False
golden_ticket = True
use_neptune = True
use_swa = True

is_training = 1

if not args.just_run:
    with open('configs/exp_configs.txt', 'a' if append_file else 'w') as f:
        # enc_in, dec_in, c_out depend on the dataset AND feature (prediction task)
        for dataset, feature in itertools.product(datasets, features):
            config0 = ''
            config0 += '--task_name long_term_forecast ' \
                       f'--is_training {is_training} ' \
                       '--delete_checkpoints '
            if use_neptune:
                config0 += '--use_neptune '
            if 'ETT' in dataset and feature == 'M':
                config0 += f'--root_path ./dataset/ETT-small ' \
                           f'--data_path {dataset}.csv ' \
                           f'--data {dataset} ' \
                           f'--feature {feature} ' \
                           f'--enc_in 7 ' \
                           f'--dec_in 7 ' \
                           f'--c_out 7 ' \
                           f'--d_model 512 '
            elif dataset == 'electricity' and feature == 'M':
                config0 += f'--root_path ./dataset/electricity ' \
                           f'--data_path {dataset}.csv ' \
                           f'--feature {feature} ' \
                           f'--enc_in 321 ' \
                           f'--dec_in 321 ' \
                           f'--c_out 321 ' \
                           f'--data custom '
            elif dataset == 'exchange_rate' and feature == 'M':
                config0 += f'--root_path ./dataset/exchange_rate ' \
                           f'--data_path {dataset}.csv ' \
                           f'--feature {feature} ' \
                           f'--enc_in 8 ' \
                           f'--dec_in 8 ' \
                           f'--c_out 8 ' \
                           f'--data custom '
            elif dataset == 'traffic' and feature == 'M':
                config0 += f'--root_path ./dataset/traffic ' \
                           f'--data_path {dataset}.csv ' \
                           f'--feature {feature} ' \
                           f'--enc_in 862 ' \
                           f'--dec_in 862 ' \
                           f'--c_out 862 ' \
                           f'--data custom '
            elif dataset == 'weather' and feature == 'M':
                config0 += f'--root_path ./dataset/weather ' \
                           f'--data_path {dataset}.csv ' \
                           f'--feature {feature} ' \
                           f'--enc_in 21 ' \
                           f'--dec_in 21 ' \
                           f'--c_out 21 ' \
                           f'--data custom '
            elif dataset == 'national_illness' and feature == 'M':
                config0 += f'--root_path ./dataset/illness ' \
                           f'--data_path {dataset}.csv ' \
                           f'--feature {feature} ' \
                           f'--enc_in 7 ' \
                           f'--dec_in 7 ' \
                           f'--c_out 7 ' \
                           f'--data custom '
            if 'ETT' in dataset and feature == 'S':
                config0 += f'--root_path ./dataset/ETT-small ' \
                           f'--data_path {dataset}.csv ' \
                           f'--data {dataset} ' \
                           f'--feature {feature} ' \
                           f'--enc_in 1 ' \
                           f'--dec_in 1 ' \
                           f'--c_out 1 ' \
                           f'--d_model 512 '
            elif dataset == 'electricity' and feature == 'S':
                config0 += f'--root_path ./dataset/electricity ' \
                           f'--data_path {dataset}.csv ' \
                           f'--feature {feature} ' \
                           f'--enc_in 1 ' \
                           f'--dec_in 1 ' \
                           f'--c_out 1 ' \
                           f'--data custom '
            elif dataset == 'exchange_rate' and feature == 'S':
                config0 += f'--root_path ./dataset/exchange_rate ' \
                           f'--data_path {dataset}.csv ' \
                           f'--feature {feature} ' \
                           f'--enc_in 1 ' \
                           f'--dec_in 1 ' \
                           f'--c_out 1 ' \
                           f'--data custom '
            elif dataset == 'traffic' and feature == 'S':
                config0 += f'--root_path ./dataset/traffic ' \
                           f'--data_path {dataset}.csv ' \
                           f'--feature {feature} ' \
                           f'--enc_in 1 ' \
                           f'--dec_in 1 ' \
                           f'--c_out 1 ' \
                           f'--data custom '
            elif dataset == 'weather' and feature == 'S':
                config0 += f'--root_path ./dataset/weather ' \
                           f'--data_path {dataset}.csv ' \
                           f'--feature {feature} ' \
                           f'--enc_in 1 ' \
                           f'--dec_in 1 ' \
                           f'--c_out 1 ' \
                           f'--data custom '
            elif dataset == 'national_illness' and feature == 'S':
                config0 += f'--root_path ./dataset/illness ' \
                           f'--data_path {dataset}.csv ' \
                           f'--feature {feature} ' \
                           f'--enc_in 1 ' \
                           f'--dec_in 1 ' \
                           f'--c_out 1 ' \
                           f'--data custom '
            pred_lengths = general_pred_lengths if dataset != 'national_illness' else ili_pred_lengths

            for model, pred_len in itertools.product(models, pred_lengths):
                if model == 'PatchTST' and dataset == 'national_illness':
                    seq_len, label_len, patience, swa_start = 104, 18, 10*100, 101
                    config1 = config0 + f'--model_id {dataset}_{seq_len}_{pred_len} ' \
                                        f'--model {model} ' \
                                        f'--seq_len {seq_len} ' \
                                        f'--label_len {label_len} ' \
                                        f'--pred_len {pred_len} ' \
                                        f'--d_model 16 ' \
                                        f'--n_heads 4 ' \
                                        f'--e_layers 3 ' \
                                        f'--d_ff 128 ' \
                                        f'--dropout 0.3 ' \
                                        f'--fc_dropout 0.3 ' \
                                        f'--batch_size 16 ' \
                                        f'--patience {patience} ' \
                                        f'--learning_rate 0.0025 ' \
                                        f'--lradj TST ' \
                                        f'--patch_len 24 ' \
                                        f'--stride 2 ' \
                                        f'--pct_start 0.2 ' \
                                        f'--d_layers 1 ' \
                                        f'--factor 1 ' \
                                        f'--des test ' \
                                        f'--itr 1 ' \
                                        f'--swa_start {swa_start} ' \
                                        f'--train_epochs 100 '
                elif model == 'PatchTST' and dataset == 'weather':
                    seq_len, label_len, patience, swa_start = 336, 48, 20*100, 101
                    config1 = config0 + f'--model_id {dataset}_{seq_len}_{pred_len} ' \
                                        f'--model {model} ' \
                                        f'--seq_len {seq_len} ' \
                                        f'--label_len {label_len} ' \
                                        f'--pred_len {pred_len} ' \
                                        f'--d_model 128 ' \
                                        f'--n_heads 16 ' \
                                        f'--e_layers 3 ' \
                                        f'--d_ff 256 ' \
                                        f'--dropout 0.2 ' \
                                        f'--fc_dropout 0.2 ' \
                                        f'--batch_size 128 ' \
                                        f'--patience {patience} ' \
                                        f'--learning_rate 0.0001 ' \
                                        f'--lradj TST ' \
                                        f'--patch_len 16 ' \
                                        f'--stride 8 ' \
                                        f'--pct_start 0.4 ' \
                                        f'--d_layers 1 ' \
                                        f'--factor 1 ' \
                                        f'--des test ' \
                                        f'--itr 1 ' \
                                        f'--swa_start {swa_start} ' \
                                        f'--train_epochs 100 '
                elif model == 'PatchTST' and dataset == 'traffic':
                    seq_len, label_len, patience, swa_start = 336, 48, 10*100, 101
                    config1 = config0 + f'--model_id {dataset}_{seq_len}_{pred_len} ' \
                                        f'--model {model} ' \
                                        f'--seq_len {seq_len} ' \
                                        f'--label_len {label_len} ' \
                                        f'--pred_len {pred_len} ' \
                                        f'--d_model 128 ' \
                                        f'--n_heads 16 ' \
                                        f'--e_layers 3 ' \
                                        f'--d_ff 256 ' \
                                        f'--dropout 0.2 ' \
                                        f'--fc_dropout 0.2 ' \
                                        f'--batch_size 24 ' \
                                        f'--patience {patience} ' \
                                        f'--learning_rate 0.0001 ' \
                                        f'--lradj TST ' \
                                        f'--patch_len 16 ' \
                                        f'--stride 8 ' \
                                        f'--pct_start 0.2 ' \
                                        f'--d_layers 1 ' \
                                        f'--factor 1 ' \
                                        f'--des test ' \
                                        f'--itr 1 ' \
                                        f'--swa_start {swa_start} ' \
                                        f'--train_epochs 100 '
                elif model == 'PatchTST' and dataset == 'electricity':
                    seq_len, label_len, patience, swa_start = 336, 48, 10*100, 101
                    config1 = config0 + f'--model_id {dataset}_{seq_len}_{pred_len} ' \
                                        f'--model {model} ' \
                                        f'--seq_len {seq_len} ' \
                                        f'--label_len {label_len} ' \
                                        f'--pred_len {pred_len} ' \
                                        f'--d_model 128 ' \
                                        f'--n_heads 16 ' \
                                        f'--e_layers 3 ' \
                                        f'--d_ff 256 ' \
                                        f'-- dropout 0.2 ' \
                                        f'--fc_dropout 0.2 ' \
                                        f'--batch_size 32 ' \
                                        f'--patience {patience} ' \
                                        f'--learning_rate 0.0001 ' \
                                        f'--lradj TST ' \
                                        f'--patch_len 16 ' \
                                        f'--stride 8 ' \
                                        f'--pct_start 0.2 ' \
                                        f'--d_layers 1 ' \
                                        f'--factor 1 ' \
                                        f'--des test ' \
                                        f'--itr 1 ' \
                                        f'--swa_start {swa_start} ' \
                                        f'--train_epochs 100 '
                elif model == 'PatchTST' and dataset in ['ETTh1', 'ETTh2', 'exchange_rate']:
                    seq_len, label_len, patience, swa_start = 336, 48, 20*100, 101
                    config1 = config0 + f'--model_id {dataset}_{seq_len}_{pred_len} ' \
                                        f'--model {model} ' \
                                        f'--seq_len {seq_len} ' \
                                        f'--label_len {label_len} ' \
                                        f'--pred_len {pred_len} ' \
                                        f'--d_model 16 ' \
                                        f'--n_heads 4 ' \
                                        f'--e_layers 3 ' \
                                        f'--d_ff 128 ' \
                                        f'--dropout 0.3 ' \
                                        f'--fc_dropout 0.3 ' \
                                        f'--batch_size 128 ' \
                                        f'--patience {patience} ' \
                                        f'--learning_rate 0.0001 ' \
                                        f'--lradj TST ' \
                                        f'--patch_len 16 ' \
                                        f'--stride 8 ' \
                                        f'--pct_start 0.3 ' \
                                        f'--d_layers 1 ' \
                                        f'--factor 1 ' \
                                        f'--des test ' \
                                        f'--itr 1 ' \
                                        f'--swa_start {swa_start} ' \
                                        f'--train_epochs 100 '
                elif model == 'PatchTST' and dataset in ['ETTm1', 'ETTm2']:
                    seq_len, label_len, patience, swa_start = 336, 48, 20*100, 101
                    config1 = config0 + f'--model_id {dataset}_{seq_len}_{pred_len} ' \
                                        f'--model {model} ' \
                                        f'--seq_len {seq_len} ' \
                                        f'--label_len {label_len} ' \
                                        f'--pred_len {pred_len} ' \
                                        f'--d_model 128 ' \
                                        f'--n_heads 16 ' \
                                        f'--e_layers 3 ' \
                                        f'--d_ff 256 ' \
                                        f'--dropout 0.2 ' \
                                        f'--fc_dropout 0.2 ' \
                                        f'--batch_size 128 ' \
                                        f'--patience {patience} ' \
                                        f'--learning_rate 0.0001 ' \
                                        f'--lradj TST ' \
                                        f'--patch_len 16 ' \
                                        f'--stride 8 ' \
                                        f'--pct_start 0.4 ' \
                                        f'--d_layers 1 ' \
                                        f'--factor 1 ' \
                                        f'--des test ' \
                                        f'--itr 1 ' \
                                        f'--swa_start {swa_start} ' \
                                        f'--train_epochs 100 '
                elif model in ['Autoformer', 'FEDformer'] and dataset == 'national_illness':
                    seq_len, label_len, patience, swa_start = 36, 18, 3*100, 101
                    config1 = config0 + f'--model_id {dataset}_{seq_len}_{pred_len} ' \
                                        f'--model {model} ' \
                                        f'--seq_len {seq_len} ' \
                                        f'--label_len {label_len} ' \
                                        f'--pred_len {pred_len} ' \
                                        f'--d_model 512 ' \
                                        f'--n_heads 8 ' \
                                        f'--e_layers 2 ' \
                                        f'--d_ff 2048 ' \
                                        f'--dropout 0.05 ' \
                                        f'--batch_size 32 ' \
                                        f'--patience {patience} ' \
                                        f'--learning_rate 0.0001 ' \
                                        f'--lradj type1 ' \
                                        f'--d_layers 1 ' \
                                        f'--factor 1 ' \
                                        f'--des test ' \
                                        f'--itr 1 ' \
                                        f'--swa_start 101 ' \
                                        f'--train_epochs 10 '
                else:
                    seq_len, label_len, patience, swa_start = 96, 48, 3*100, 101
                    config1 = config0 + f'--model {model} ' \
                                        f'--seq_len {seq_len} ' \
                                        f'--label_len {label_len} ' \
                                        f'--pred_len {pred_len} ' \
                                        f'--d_model 512 ' \
                                        f'--n_heads 8 ' \
                                        f'--e_layers 2 ' \
                                        f'--d_ff 2048 ' \
                                        f'--dropout 0.05 ' \
                                        f'--batch_size 32 ' \
                                        f'--patience {patience} ' \
                                        f'--learning_rate 0.0001 ' \
                                        f'--lradj type1 ' \
                                        f'--d_layers 1 ' \
                                        f'--factor 1 ' \
                                        f'--des test ' \
                                        f'--itr 1 ' \
                                        f'--swa_start {swa_start} ' \
                                        f'--train_epochs 10 '
                if use_swa:
                    patience = 101
                    custom_averaging_flags = [0]
                    anneal_strategies = ['cos']
                    swa_lr = [1e-3, 1e-4, 1e-5] #1e-3, 1e-4, 1e-5

                    anneal_epochs = [1, 2, 3] #1, 2, 3
                    swa_starts = [3] #3, 5, 6, 8
                    factor = 10 if model == 'PatchTST' else 1
                    anneal_epochs = [element*factor for element in anneal_epochs]
                    swa_starts = [element*factor for element in swa_starts]

                    for strategy, anneal_epoch, lr, flag, swa_start in itertools.product(anneal_strategies, anneal_epochs, swa_lr,
                                                                                         custom_averaging_flags, swa_starts):
                        config2 = config1 + f'--anneal_strategy {strategy} ' \
                                            f'--anneal_epoch {anneal_epoch} ' \
                                            f'--swa_lr {lr} ' \
                                            f'--custom_averaging {flag} ' \
                                            f'--swa_start {swa_start} ' \
                                            f'--patience {patience} '


                        # Split the string into individual configuration options
                        configs_list = config2.split()

                        # Create a dictionary to store the last occurrence of each configuration option
                        unique_configs = {}

                        # Iterate through the list of configuration options and update the dictionary
                        current_option = None
                        for option in configs_list:
                            if option.startswith('--'):
                                if current_option is not None:
                                    unique_configs[current_option] = None  # Assign None for options without values
                                current_option = option
                            elif current_option is not None:
                                unique_configs[current_option] = option
                                current_option = None

                        # Join the unique configuration options back into a string
                        new_configs = ' '.join(
                            [f"{key} {value}" if value is not None else key for key, value in unique_configs.items()])


                        f.write(new_configs + '\n')
                else:
                    f.write(config1 + '\n')

# logs dir with the current date and time
logs_dir = 'swa_logs/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

if golden_ticket:
    partition = 'rtx6000'
    qos = 'azencot'
else:
    partition = 'main'
    qos = 'normal'

# calculate array size from the length of the config file
config_file = 'configs/exp_configs.txt'
with open(config_file, 'r') as f:
    array_size = len(f.readlines())

qos = 'azencot' if golden_ticket else 'normal'

file = '/cs_storage/yuvalao/code/Time-Series-Library-main/run.py'

# create the slurm file
sbatch = f"""#!/bin/bash

#SBATCH --chdir=/cs_storage/yuvalao/code/Time-Series-Library-main


#SBATCH --time 7-00:00:00       ### Job running time limit. Make sure it is not exceeding the partition time limit! Format: D-H:MM:SS
#SBATCH --job-name 'swa_run'     ### Name of the job. replace my_job with your desired job name
#SBATCH --output /home/yuvalao/logs/{logs_dir}/swa_job-%J.out     ### stdout log for running job - %J is the job number variable
#SBATCH --error /home/yuvalao/logs/{logs_dir}/swa_job-%J.err      ### stderr log for running job
#SBATCH --mail-user=yuvalao@post.bgu.ac.il    ### User's email for sending job status
#SBATCH --mail-type=ALL        ### Conditions when to send the email. ALL,BEGIN,END,FAIL, REQUEU, NONE

#SBATCH --gpus=rtx_6000:1            ### number of GPUs, ask for more than 1 only if you can parallelize your code for multi GPU
#SBATCH --partition={partition}          ### golden ticket babyyyyyy
#SBATCH --array=1-{array_size}
#SBATCH --qos={qos}          ### golden ticket babyyyyyy
#SBATCH --mem=32G              ### ammount of RAM memory
#SBATCH --cpus-per-task=6     ### number of CPU cores

### Print some data to output file ###
echo `date`
echo -e "\\nSLURM_JOBID:\\t\\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\\t" $SLURM_JOB_NODELIST "\\n\\n"

### Start you code below ####
module load anaconda            ### load anaconda module (must present when working with conda environments)
source activate swa         ### activating environment, environment must be configured before running the job
config=`awk "NR==$SLURM_ARRAY_TASK_ID{{ print }}" {config_file}`
echo $config
python -u {file} $config
echo 'done'
echo `date`
"""

if not just_generate:
    if not os.path.exists(f'/home/yuvalao/logs/{logs_dir}'):
        os.makedirs(f'/home/yuvalao/logs/{logs_dir}')

    with open(f'configs/run.sbatch', 'w') as f:
        f.write(sbatch)

    # run the slurm file
    os.system(f'sbatch configs/run.sbatch')

    # delete the slurm file
    os.system(f'rm configs/run.sbatch')
