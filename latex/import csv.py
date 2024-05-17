#%%
from neptune import management
from tqdm import tqdm
import pandas as pd
import itertools
import neptune
import torch
import tqdm
import csv
import sys
import os
#%%
def define_exp_params():
    """
    All experiments.

    Returns 4 lists of strings: models, datasets, features, general_pred_lengths.
    """
    models = ['Autoformer', 'FEDformer', 'PatchTST']
    datasets = ['ETTm2', 'weather', 'national_illness'] #{'traffic', 'electricity', 'exchange_rate', 'national_illness']
    features = ['S', 'M']
    general_pred_lengths = [96, 192, 336, 720]

    return models, datasets, features, general_pred_lengths

def get_setting_key(models, datasets, features, preds):
    """
    Creates a generator for experiment settings, all should apper in results files.

    models: model name, options: [Autoformer, FEDformer, PatchTST]'
    datasets: dataset name, options: [ETTm2, weather, traffic, electricity, exchange_rate, national_illness]
    features: forecasting task, options: [M, S]
    preds: prediction sequence length, options: [96, 192, 336, 720]
    """
    preds_to_ILI_preds = {'96': 24, '192': 36, '336': 48, '720': 60}

    for model, dataset, feature, pred in itertools.product(models, datasets, features, preds):
        if dataset == 'national_illness':
            dataset = 'ILI'
            pred = preds_to_ILI_preds[str(pred)]

        yield f'{model},{dataset},{pred},{feature}'

def ids_for_setting(file_name, setting):
    """
    file_name: original or swa
    setting: target experiment setting (to look for within file location).

    Returns 'seed_counts' dictionary with:
        a). count of runs for each 'seed_itr'
        b). id of the latest run (to keep)
        c). list of the remaining ids for that run (to trash)
    """
    file_location = f'/cs_storage/yuvalao/code/latex/results_{file_name}.csv' if 'levy3' not in sys.executable else f'C:/Users/levy3/OneDrive/שולחן העבודה/Work/Thesis/code/latex/results_{file_name}.csv'
    seed_counts = {'2021_0': {'count': 0, 'max_id': None, 'ids_for_trash': []},
                   '2021_1': {'count': 0, 'max_id': None, 'ids_for_trash': []},
                   '2021_2': {'count': 0, 'max_id': None}, 'ids_for_trash': []}

    with open(file_location, 'r') as file:
        # loop through all settings in the file
        for line in file:
            parts = line.strip().split(',')
            key = f"{parts[2]},{parts[3]},{parts[4]},{parts[5]}"

            # key matches target setting
            if setting == key:
                neptune_id = int(parts[0])
                seed = parts[1]

                # update max_id
                if seed_counts[seed]['max_id'] is None:
                    seed_counts[seed]['max_id'] = neptune_id

                # update ids_for_trash list
                if neptune_id > seed_counts[seed]['max_id']:
                    seed_counts[seed]['ids_for_trash'].append('SWA-'+str(seed_counts[seed]['max_id']))
                    seed_counts[seed]['max_id'] = neptune_id

                # update count
                seed_counts[seed]['count'] += 1

    return seed_counts

def unified_id_list_for_trash():
    """
    Creates a unified list of all ids_for_trash from both original and swa files, for all experiments.
    Returns: unified list of ids for trash
    """
    ids_for_trash = []

    # loop through both original and swa files
    for file_name in tqdm(['original', 'swa']):
        print(f'\n{file_name}')
        models, datasets, features, general_pred_lengths = define_exp_params()

        # each iteration of the loop yield a different experiment setting target to match
        for idx, key in enumerate(get_setting_key(models, datasets, features, general_pred_lengths)):
            print(f'\n{idx}: {key}')
            seed_counts = ids_for_setting(file_name, key)
            print(seed_counts)

            # append all ids_for_trash to the unified list
            for attr in seed_counts.values():
                if 'ids_for_trash' in attr:
                    ids_for_trash.extend(attr['ids_for_trash'])

    return ids_for_trash

def trash_neptune_runs(ids_for_trash):
    """
    Trashes neptuneAI failed runs and duplicates (list received as input).
    ids_for_trash: list of all duplicates to trash (from local reposit.)
    """
    project_name, token = 'azencot-group/SWA', 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyZGE5Y2Y1ZC1iZTgxLTQwN2QtOTE1OC1lMGU0NWE0ZWRmMjQifQ=='

    # fetch the id list of all failed runs from neptune
    project = neptune.init_project(project=project_name, api_token=token)
    runs_table_df = project.fetch_runs_table(state="inactive").to_pandas()
    failed_run_id = runs_table_df[runs_table_df["sys/failed"] == True]["sys/id"].values.tolist()

    # there are both failed runs and duplicate runs to trash
    if len(failed_run_id)>=1 and ids_for_trash is not None:
        failed_run_id.extend(ids_for_trash)
        print(f'Failed runs trashed: {failed_run_id}')
        management.trash_objects(project=project_name, api_token=token, ids=ids_for_trash)

    # there are only failed runs to trash
    elif len(failed_run_id)>=1 and ids_for_trash is None:
        print(f'Failed runs trashed: {failed_run_id}')
        management.trash_objects(project=project_name, api_token=token, ids=failed_run_id)

    # there are only duplicate runs to trash
    elif len(failed_run_id) < 1 and ids_for_trash is not None:
        failed_run_id.extend(ids_for_trash)
        print(f'runs trashed: {ids_for_trash}')
        management.trash_objects(project=project_name, api_token=token, ids=ids_for_trash)

    else:
        print('No runs to trash')

def write_cleaned_file():
    """Creates a cleaned version for both original and swa files."""
    for file_name in tqdm(['original', 'swa']):
        print(f'\n{file_name}')
        output_name = f'results_{file_name}2.csv'
        file_location = f'/cs_storage/yuvalao/code/latex/results_{file_name}.csv' if 'levy3' not in sys.executable else f'C:/Users/levy3/OneDrive/שולחן העבודה/Work/Thesis/code/latex/results_{file_name}.csv'

        #ids_for_trash = create_ids_for_trash()
        with open(file_location, 'r') as inp, open(output_name, 'w', newline='') as out:
            writer = csv.writer(out)
            for row in csv.reader(inp):

                # 7700 is the first id in neptune
                if int(row[0]) >= 7700:
                #if 'SWA-'+str(row[0]) not in ids_for_trash:
                    writer.writerow(row)
#%% - Activate previous cell
file_location = 'C:/Users/levy3/OneDrive/שולחן העבודה/Work/Thesis/code/latex/results_swa.csv'
duplicate_ids = unified_id_list_for_trash()
# input = open(file_location, 'rb')
# output = open('results_swa1.csv', 'wb')
with open(file_location, 'r') as inp, open('results_swa2.csv', 'w', newline='') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if 'SWA-'+str(row[0]) not in duplicate_ids:
            writer.writerow(row)

trash_neptune_runs(ids_for_trash=duplicate_ids)

project = neptune.init_project(project='azencot-group/SWA',
                               api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyZGE5Y2Y1ZC1iZTgxLTQwN2QtOTE1OC1lMGU0NWE0ZWRmMjQifQ==')
neptune_all=project.fetch_runs_table().to_pandas()['sys/id'].tolist()
write_cleaned_file()
#%%
project = neptune.init_project(project='azencot-group/SWA',
                               api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyZGE5Y2Y1ZC1iZTgxLTQwN2QtOTE1OC1lMGU0NWE0ZWRmMjQifQ==')

neptune_all=project.fetch_runs_table().to_pandas()['sys/id'].tolist()

filters = [element for element in neptune_all if int(element.split("-")[1])>7990]
management.trash_objects(project='azencot-group/SWA',
                         api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyZGE5Y2Y1ZC1iZTgxLTQwN2QtOTE1OC1lMGU0NWE0ZWRmMjQifQ==',
                         ids=filters)
# %% - Import tables
def import_tables(retreive_results=False):
    """
    Imports original and swa results files.
    retreive_results: True for results tables, False for runs tables.
    """
    for file in ['original', 'swa_lr0.0001_ae3_t6']:
        file_location = f'/cs_storage/yuvalao/code/latex/results_{file}.csv' if 'levy3' not in sys.executable else f'C:/Users/levy3/OneDrive/שולחן העבודה/Work/Thesis/code/latex/results_{file}.csv'
        
        # partition each loaded file to multivariate and univariate forecasting
        if file == 'original':
            df = pd.read_csv(file_location)
            df.columns = ['Neptune', 'Seed', 'Model', 'Dataset', 'Horizon', 'Type', 'MSE', 'MAE']
            multivariate_regular = df[df['Type'] == 'M']
            univariate_regular = df[df['Type'] == 'S']
        else:
            df_swa = pd.read_csv(file_location)
            df_swa.columns = ['Neptune_swa', 'Seed', 'Model', 'Dataset', 'Horizon', 'Type', 'MSE_swa', 'MAE_swa']
            multivariate_swa = df_swa[df_swa['Type'] == 'M']
            univariate_swa = df_swa[df_swa['Type'] == 'S']
    
    # merged tables according to forecasting task
    multi, uni = multivariate_regular.merge(multivariate_swa, on=['Seed', 'Model', 'Dataset', 'Horizon', 'Type']), univariate_regular.merge(univariate_swa, on=['Seed', 'Model', 'Dataset', 'Horizon', 'Type'])
    
    # summary tables for each forecasting task
    multi_runs, uni_runs = multi[['Seed', 'Model', 'Dataset', 'Horizon', 'Type', 'Neptune', 'Neptune_swa']], uni[['Seed', 'Model', 'Dataset', 'Horizon', 'Type', 'Neptune', 'Neptune_swa']]
    # results tables for each forecasting type
    multi_results, uni_results = multi[['Seed', 'Model', 'Dataset', 'Horizon', 'Type', 'MSE', 'MSE_swa', 'MAE', 'MAE_swa']], uni[['Seed', 'Model', 'Dataset', 'Horizon', 'Type', 'MSE', 'MSE_swa', 'MAE', 'MAE_swa']]
    multi_results, uni_results = multi.groupby(['Model', 'Dataset', 'Horizon'])[['MSE', 'MSE_swa', 'MAE', 'MAE_swa']].aggregate(['mean', 'std']).round(4), uni.groupby(['Model', 'Dataset', 'Horizon'])[['MSE', 'MSE_swa', 'MAE', 'MAE_swa']].aggregate(['mean', 'std']).round(4)
    
    if retreive_results:
        return multi_results, uni_results
    else:
        return multi_runs, uni_runs


def swa_vs_original_results(stat_significant=False, multi=True, mse_improved=True, mae_improved=True):
    """returns a list of experiments that meet the provided conditions.

    args:
        stat_significant (bool): True for ammending mean values of swa to also include its std.
        multi (bool): True for multivariate forecasting, False for univariate.
        mse_improved (bool): True in case SWA improves MSE over the original run.
        mae_improved (bool): True in case SWA improves MAE over the original run.
    """
    multi_results, uni_results = import_tables(True)
    if multi:
        print(f'\t--Multivariate forecasting--')
        df = multi_results
    
    else:
        print(f'\t--Univariate forecasting--')
        df = uni_results
    
    
    mae_original_mean, mse_original_mean = df.loc[:, ('MAE', 'mean')], df.loc[:, ('MSE', 'mean')]
    mae_swa_mean, mse_swa_mean = df.loc[:, ('MAE_swa', 'mean')], df.loc[:, ('MSE_swa', 'mean')]
        
    if mse_improved and mae_improved:
        print(f'SWA improves BOTH MSE & MAE in the following experiment settings:')
        
        if stat_significant:
            print(f'--"statistical significance":')
            print(f'MSE_SWA_\u03BC+MSE_SWA_\u03C3')
            mse_swa_mean = df.loc[:, ('MSE_swa', 'mean')] + df.loc[:, ('MSE_swa', 'std')]
            print(f'MAE_SWA_\u03BC+MAE_SWA_\u03C3--\n')
            mae_swa_mean = df.loc[:, ('MAE_swa', 'mean')] + df.loc[:, ('MAE_swa', 'std')]
        else:
            print(f'regular scenario\n')
        
        return df[(mse_original_mean > mse_swa_mean) & (mae_original_mean > mae_swa_mean)].index.tolist()
        
        
    elif mse_improved and not mae_improved:
        print(f'SWA improves MSE, but worsens MAE in the following experiment settings:')
        
        if stat_significant:
            print(f'--"statistical significance":')
            print(f'MSE_SWA_\u03BC+MSE_SWA_\u03C3')
            mse_swa_mean = df.loc[:, ('MSE_swa', 'mean')] + df.loc[:, ('MSE_swa', 'std')]
            print(f'MAE_SWA_\u03BC-MAE_SWA_\u03C3--\n')
            mae_swa_mean = df.loc[:, ('MAE_swa', 'mean')] - df.loc[:, ('MAE_swa', 'std')]
        else:
            print(f'regular scenario\n')
            
        return df[(mse_original_mean > mse_swa_mean) & (mae_original_mean <= mae_swa_mean)].index.tolist()
        
        
    elif not mse_improved and mae_improved:
        print(f'SWA worsens MSE, but improves MAE in the following experiment settings:')
        
        if stat_significant:
            print(f'--"statistical significance":')
            print(f'MSE_SWA_\u03BC-MSE_SWA_\u03C3')
            mse_swa_mean = df.loc[:, ('MSE_swa', 'mean')] - df.loc[:, ('MSE_swa', 'std')]
            print(f'MAE_SWA_\u03BC+MAE_SWA_\u03C3--\n')
            mae_swa_mean = df.loc[:, ('MAE_swa', 'mean')] + df.loc[:, ('MAE_swa', 'std')]
        else:
            print(f'regular scenario\n')
            
        return df[(mse_original_mean <= mse_swa_mean) & (mae_original_mean > mae_swa_mean)].index.tolist()
        
    
    else:
        print(f'\nSWA worsens BOTH MSE & MAE in the following experiment settings:')
        
        if stat_significant:
            print(f'--"statistical significance":')
            print(f'MSE_SWA_\u03BC-MSE_SWA_\u03C3')
            mse_swa_mean = df.loc[:, ('MSE_swa', 'mean')] - df.loc[:, ('MSE_swa', 'std')]
            print(f'MAE_SWA_\u03BC-MAE_SWA_\u03C3--\n')
            mae_swa_mean = df.loc[:, ('MAE_swa', 'mean')] - df.loc[:, ('MAE_swa', 'std')]
        else:
            print(f'regular scenario\n')
        
        
        return df[(mse_original_mean <= mse_swa_mean) & (mae_original_mean <= mae_swa_mean)].index.tolist()
# %% 
def swa_preformace_rates(scenario_type='regular', multi=True):
    """
    Calculates the deterioration rates of SWA over the original run.

    scenario_type: str, options:
        ####stat - "statistical significant". Even when (swa_mu-swa_std), SWA >= Original.
        worst - widest gap possible between swa and original results: (swa_mu+swa_std) and (original_mu-original_std).
        regular - std not taken into account.
        best - smallest gap possible between swa and original results: (swa_mu-swa_std) and (original_mu+original_std).
    multi: bool, True for multivariate forecasting, False for univariate.
    """
    multi_results, uni_results = import_tables(True)
    assert scenario_type is not None, 'scenario_type must be one of the following: [best, regular, worst]'
    
    if multi:
        print(f'\t--SWA preformance on Multivariate forecasting--')
        
        # if scenario_type == 'stat':
        #     print(f'\t--SWA_\u03C3 taken into account--\n')
            
        #     # update mae_swa_mean (alone) according to stat scenario
        #     mae_swa_mean = multi_results.loc[:, ('MAE_swa', 'mean')] + multi_results.loc[:, ('MAE_swa', 'std')]
        #     mae_original_mean = multi_results.loc[:, ('MAE', 'mean')]
            
        #     # update mse_swa_mean (alone) according to stat scenario
        #     mse_swa_mean = multi_results.loc[:, ('MSE_swa', 'mean')] + multi_results.loc[:, ('MSE_swa', 'std')]
        #     mse_original_mean = multi_results.loc[:, ('MSE', 'mean')]
        
        
        if scenario_type == 'worst':
            print(f'\t--updating swa_\u03BC+swa_\u03C3 and original_\u03BC-original_\u03C3--\n')
            
            # update mae_swa_mean and mae_original_mean according to best scenario
            mae_swa_mean = multi_results.loc[:, ('MAE_swa', 'mean')] + multi_results.loc[:, ('MAE_swa', 'std')]
            mae_original_mean = multi_results.loc[:, ('MAE', 'mean')] - multi_results.loc[:, ('MAE', 'std')]
            
            # update mse_swa_mean and mse_original_mean according to best scenario
            mse_swa_mean = multi_results.loc[:, ('MSE_swa', 'mean')] + multi_results.loc[:, ('MSE_swa', 'std')]
            mse_original_mean = multi_results.loc[:, ('MSE', 'mean')] - multi_results.loc[:, ('MSE', 'std')]
        
        
        elif scenario_type == 'best':
            print(f'\t--updating swa_\u03BC-swa_\u03C3 and original_\u03BC+original_\u03C3--\n')
            
            # update mae_swa_mean and mae_original_mean according to worst scenario
            mae_swa_mean = multi_results.loc[:, ('MAE_swa', 'mean')] - multi_results.loc[:, ('MAE_swa', 'std')]
            mae_original_mean = multi_results.loc[:, ('MAE', 'mean')] + multi_results.loc[:, ('MAE', 'std')]
            
            # update mse_swa_mean and mse_original_mean according to worst scenario
            mse_swa_mean = multi_results.loc[:, ('MSE_swa', 'mean')] - multi_results.loc[:, ('MSE_swa', 'std')]
            mse_original_mean = multi_results.loc[:, ('MSE', 'mean')] + multi_results.loc[:, ('MSE', 'std')]
        
        
        else:
            print(f'\t\t--\u03C3 not taken into account--\n')
            
            # update mae_swa_mean and mae_original_mean according to regular scenario
            mae_swa_mean = multi_results.loc[:, ('MAE_swa', 'mean')]
            mae_original_mean = multi_results.loc[:, ('MAE', 'mean')]
            
            # update mse_swa_mean and mse_original_mean according to regular scenario
            mse_swa_mean = multi_results.loc[:, ('MSE_swa', 'mean')]
            mse_original_mean = multi_results.loc[:, ('MSE', 'mean')]
        
        
        mae_deterioration = (mae_swa_mean - mae_original_mean) / mae_original_mean * 100
        mae_deterioration = pd.DataFrame(mae_deterioration, columns=['MAE deterioration (%)'])
        
        mse_deterioration = (mse_swa_mean - mse_original_mean) / mse_original_mean * 100
        mse_deterioration = pd.DataFrame(mse_deterioration, columns=['MSE deterioration (%)'])
        
        df = mse_deterioration.merge(mae_deterioration, on=['Model', 'Dataset', 'Horizon'])
        
        return df.reset_index(level=[0, 1, 2]).sort_values(by='MSE deterioration (%)', ascending=False)
    
    
    else:
        print(f'\t--SWA preformance on Univariate forecasting--')
        
        if scenario_type == 'stat':
            print(f'\t--SWA_\u03C3 taken into account--\n')
            
            # update mae_swa_mean (alone) according to stat scenario
            mae_swa_mean = uni_results.loc[:, ('MAE_swa', 'mean')] + uni_results.loc[:, ('MAE_swa', 'std')]
            mae_original_mean = uni_results.loc[:, ('MAE', 'mean')]
            
            # update mse_swa_mean (alone) according to stat scenario
            mse_swa_mean = uni_results.loc[:, ('MSE_swa', 'mean')] + uni_results.loc[:, ('MSE_swa', 'std')]
            mse_original_mean = uni_results.loc[:, ('MSE', 'mean')]
            
            
        elif scenario_type == 'worst':
            print(f'\t--updating swa_\u03BC+swa_\u03C3 and original_\u03BC-original_\u03C3--\n')
            
            # update mae_swa_mean and mae_original_mean according to best scenario
            mae_swa_mean = uni_results.loc[:, ('MAE_swa', 'mean')] + uni_results.loc[:, ('MAE_swa', 'std')]
            mae_original_mean = uni_results.loc[:, ('MAE', 'mean')] - uni_results.loc[:, ('MAE', 'std')]
            
            # update mse_swa_mean and mse_original_mean according to best scenario
            mse_swa_mean = uni_results.loc[:, ('MSE_swa', 'mean')] + uni_results.loc[:, ('MSE_swa', 'std')]
            mse_original_mean = uni_results.loc[:, ('MSE', 'mean')] - uni_results.loc[:, ('MSE', 'std')]

        
        elif scenario_type == 'best':
            print(f'\t--updating swa_\u03BC-swa_\u03C3 and original_\u03BC+original_\u03C3--\n')
            
            # update mae_swa_mean and mae_original_mean according to worst scenario
            mae_swa_mean = uni_results.loc[:, ('MAE_swa', 'mean')] - uni_results.loc[:, ('MAE_swa', 'std')]
            mae_original_mean = uni_results.loc[:, ('MAE', 'mean')] + uni_results.loc[:, ('MAE', 'std')]
            
            # update mse_swa_mean and mse_original_mean according to worst scenario
            mse_swa_mean = uni_results.loc[:, ('MSE_swa', 'mean')] - uni_results.loc[:, ('MSE_swa', 'std')]
            mse_original_mean = uni_results.loc[:, ('MSE', 'mean')] + uni_results.loc[:, ('MSE', 'std')]

        
        else:
            print(f'\t\t--\u03C3 not taken into account--\n')
            
            # update mae_swa_mean and mae_original_mean according to regular scenario
            mae_swa_mean = uni_results.loc[:, ('MAE_swa', 'mean')]
            mae_original_mean = uni_results.loc[:, ('MAE', 'mean')]
            
            # update mse_swa_mean and mse_original_mean according to regular scenario
            mse_swa_mean = uni_results.loc[:, ('MSE_swa', 'mean')]
            mse_original_mean = uni_results.loc[:, ('MSE', 'mean')]
        
        
        mae_deterioration = (mae_swa_mean - mae_original_mean) / mae_original_mean * 100
        mae_deterioration = pd.DataFrame(mae_deterioration, columns=['MAE deterioration (%)'])
        
        mse_deterioration = (mse_swa_mean - mse_original_mean) / mse_original_mean * 100
        mse_deterioration = pd.DataFrame(mse_deterioration, columns=['MSE deterioration (%)'])
        
        df = mse_deterioration.merge(mae_deterioration, on=['Model', 'Dataset', 'Horizon'])
        
        return df.reset_index(level=[0, 1, 2]).sort_values(by='MSE deterioration (%)', ascending=False)
# %%
project_name, token = 'azencot-group/SWA', 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyZGE5Y2Y1ZC1iZTgxLTQwN2QtOTE1OC1lMGU0NWE0ZWRmMjQifQ=='
project = neptune.init_project(project=project_name, api_token=token)
inactives_df = project.fetch_runs_table(state="inactive").to_pandas()
failed_df = inactives_df[(inactives_df["sys/failed"] == True) & (inactives_df['sys/id'].str.split('-').str[1].astype(int) > 7990)]
runs = failed_df[['model/parameters/model', 'model/parameters/data', 'model/parameters/features',
                  'model/parameters/pred_len', 'model/parameters/swa_start',
                  'model/parameters/swa_lr', 'model/parameters/anneal_epochs']]
#%%
def writeToMissingExpFile(model, dataset, horizon, type, start, lr, ae):
    with (open('missing_exp.txt', 'a') as outPutFile):
        with open('/cs_storage/yuvalao/code/Time-Series-Library-main/configs/exp_configs.txt', 'r') as input_file:
            for line in input_file:
                if (model in line) & (dataset in line) & (str(horizon) in line) & (type in line) & (str(start) in line) & (str(lr) in line) & (str(ae) in line):
                    outPutFile.write(line)
                    break

df = pd.read_csv('/cs_storage/yuvalao/code/latex/results_swa.csv')
df.columns = ['Neptune', 'Seed', 'Model', 'Dataset', 'Horizon', 'Type', 'SWA_start', 'SWA_ae', 'SWA_lr', 'MSE', 'MAE']

df = df.iloc[:, 2:-2]
df = df.drop_duplicates()

models = ['Autoformer', 'FEDformer', 'PatchTST']
datasets = ['ETTm2', 'weather']
horizons = [96, 192, 336, 720]
types = ['S', 'M']

missing = 0
complete = 0
for model, dataset, horizon, type in itertools.product(models, datasets, horizons, types):
    startings = [3, 5, 6, 8] if model != 'PatchTST' else [30, 50, 60, 80]
    learning_rates = [1e-3, 1e-4, 1e-5]
    anneal_epochs = [1, 2, 3] if model != 'PatchTST' else [10, 20, 30]

    for start, lr, ae in itertools.product(startings, learning_rates, anneal_epochs):
        run = f"{start},{ae},{lr}"
        exp = f"{model},{dataset},{horizon},{type}"

        temp = df[(df['Model'] == model) & (df['Dataset'] == dataset) & (df['Horizon'] == horizon) &
                  (df['Type'] == type) & (df['SWA_start'] == start) & (df['SWA_lr'] == lr) & (df['SWA_ae'] == ae)]

        if temp.shape[0] == 0:
            missing += 1
            print(f'{run} missing for {exp}')
            #writeToMissingExpFile(model, dataset, horizon, type, start, lr, ae)
        else:
            complete += 1

print(f'\n{missing} missing, {complete} complete, total: {missing + complete}')

# with open('/cs_storage/yuvalao/code/Time-Series-Library-main/configs/exp_configs.txt', 'r') as input_file:
#     with open('/cs_storage/yuvalao/code/Time-Series-Library-main/configs/exp_configs1.txt', 'w') as output_file:
#         for line in input_file:
#             # if line in
#             output_file.write(line)