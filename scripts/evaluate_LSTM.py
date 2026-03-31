""" evaluate performance of LSTM deep learning model """

import torch
from torch.utils.data import DataLoader
import pandas as pd
import pickle as pkl
from sklearn.metrics import root_mean_squared_error
import numpy as np
import json
from paths import DATA_PATH, MODELS_PATH, CONFIG_PATH, LOGS_PATH
import os
import yaml
from collections import defaultdict
from classes import LSTMNoEmbed, SeqExtractionDataSet


def unscale_per_client(group, dict_with_mean_and_std):
    """ use mean and std usage for each client to unscale data, ready for 
    evaluation """

    client_id = int(np.unique(group.index.get_level_values('client_id'))[0])

    return ((group * dict_with_mean_and_std[client_id]['std']) + dict_with_mean_and_std[client_id]['mean'])


def log_stats(model_name: str, nrmse_per_client, nrmse_summary_dict, preds_dict):
    """ log performance metrics acrosss clients along with summary stats of these, and 
    the predictions for future visualisation """

    # define path to save and check that exists
    save_path = LOGS_PATH / model_name
    os.makedirs(save_path, exist_ok=True)

    with open(save_path / 'nrmse_summary_stats.json', 'w') as f:
        json.dump(nrmse_summary_dict, f, indent=4)

    with open(save_path / 'nrmse_per_client.json', 'w') as f:
        # convert to list so serialisable
        json.dump(nrmse_per_client.to_list(), f, indent=4)
    
    # save predictions (only) for further evaluation/visualisation (also gives client ids used)
    os.makedirs(DATA_PATH / 'processed', exist_ok=True)
    with open(DATA_PATH / 'processed' / f'{model_name}_preds_dict.pkl', 'wb') as f:
        pkl.dump(preds_dict, f)



def evaluate_LSTM():
    
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
        LSTM_params = config['LSTM_params']

    SEQ_LENGTH = LSTM_params['seq_length']  # length of context window
    HIDDEN_SIZE = LSTM_params['hidden_size'] # hidden size in LSTM
    BATCH_SIZE = LSTM_params['batch_size']  # size of batches used during training
    MODEL_NAME = 'LSTM'  # just one model to evaluate here

    ##############
    # load in data dictionary and index map for testing data
    ##############

    with open(DATA_PATH / 'processed' / 'seq_data_dict_test', 'rb') as f:
        data_dict = pkl.load(f)

    with open(DATA_PATH / 'processed' / 'idx_map_test', 'rb') as f:
        idx_map = pkl.load(f)

    ############
    # load in mean and std usages for each client, for unscaling data
    ############

    # for unscaling labels and predictions
    with open(DATA_PATH / 'processed' / 'mean_std_per_client.json', 'r') as f:
        df_std_mean_usage = pd.read_json(f).T
        # create dictionary version for fast lookup
        dict_std_mean_usage = df_std_mean_usage.to_dict(orient='index')

    ############
    # instantiate model with trained weights and biases
    ############

    # get number of columns from training data (using data from data dict)
    n_features = data_dict[idx_map[0][0]].shape[1]

    # set up model instance
    model = LSTMNoEmbed(n_features=n_features, hidden_size=HIDDEN_SIZE)

    # and load weights and biases
    with open(MODELS_PATH / 'LSTM.pkl', 'rb') as f:
        model_state_dict = pkl.load(f)
        model.load_state_dict(model_state_dict)

    ##########
    # set up dataset and dataloader
    ##########
    
    dataset_test = SeqExtractionDataSet(data_dict=data_dict, idx_map=idx_map, SEQ_LENGTH=SEQ_LENGTH)
    # don't shuffle the testing data
    dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    ############
    # EVALUATION of testing data
    ############

    model.eval()

    # create dictionaries to store the predictions and corresponding labels for each client embedding id
    preds_dict = defaultdict(list)
    labels_dict = defaultdict(list)

    for batch_idx, (X_seqs_test, labels_test, client_ids_test) in enumerate(dataloader_test):
        
        # run the sequence and client id through the LSTM
        y_preds = model(X_seqs_test)

        # get unique embedding index of clients in each batch
        clients_ids_test_unique = client_ids_test.unique().detach().numpy()

        # batches may have overlap of client ids, so split the data across batch idx dimension if that happens
        # if only one client, then append to that clients dictionar
        if len(clients_ids_test_unique) == 1:

            # convert to int for indexing into dictionary (index element to avoid error in future numpy)
            clients_ids_test_unique = int(clients_ids_test_unique[0])

            # assign to dictionary
            preds_dict[clients_ids_test_unique] += list(y_preds.detach().numpy().flatten())
            labels_dict[clients_ids_test_unique] += list(labels_test.detach().numpy().flatten())
        elif len(clients_ids_test_unique) == 2:

            # need to split the data where client id changes
            # get the index of first instance of other client id
            index_of_change = np.where(client_ids_test == clients_ids_test_unique[1])[0][0]
            # then use to split data and save to dictionary
            preds_dict[clients_ids_test_unique[0]] += list(y_preds[:index_of_change].detach().numpy().flatten())
            labels_dict[clients_ids_test_unique[0]] += list(labels_test[:index_of_change].detach().numpy().flatten())

            preds_dict[clients_ids_test_unique[1]] += list(y_preds[index_of_change:].detach().numpy().flatten())
            labels_dict[clients_ids_test_unique[1]] += list(labels_test[index_of_change:].detach().numpy().flatten())

        else:
            raise Exception(f'incompatible number of client ids within the batch, totalling {len(clients_ids_test_unique)}')


    # for each client, get scaled labels and predictions (using their mean and std usage)
    preds_dict_unscaled = {client_id: np.array(preds_dict[client_id]) * 
                        dict_std_mean_usage[client_id]['std'] + dict_std_mean_usage[client_id]['mean'] for client_id in preds_dict.keys()}
    labels_dict_unscaled = {client_id: np.array(labels_dict[client_id]) * 
                            dict_std_mean_usage[client_id]['std'] + dict_std_mean_usage[client_id]['mean'] for client_id in labels_dict.keys()}

    # and compute the mean usage for each client
    mean_usages = {client_id: labels_dict_unscaled[client_id].mean() for client_id in labels_dict.keys()}

    # and the root mean squared error normalised by the mean test data usage
    nrmse_dict = {int(client_id): float(root_mean_squared_error(preds_dict_unscaled[client_id], 
                                                                labels_dict_unscaled[client_id])/mean_usages[client_id]) for client_id in preds_dict.keys()}

    # get values of nrmse for summary stats
    nrmse_values = list(nrmse_dict.values())
    # get clients sorted in ascending order of nrmse, for summary stats
    sorted_clients_by_nrmse = sorted(list(nrmse_dict.keys()), key=lambda x: nrmse_dict[x])

    summary_dict = {
    'mean_nrmse': np.mean(nrmse_values),
    'std_nrmse': np.std(nrmse_values),
    'max_nrmse': np.max(nrmse_values),
    'min_nrmse': np.min(nrmse_values),
    'top_5_performing_clients': sorted_clients_by_nrmse[:5],
    'bottom_5_performing_clients': sorted_clients_by_nrmse[-5:],
    }

    # log the summary stats and per client nrmse
    log_stats(model_name=MODEL_NAME, nrmse_per_client=nrmse_dict, nrmse_summary_dict=summary_dict, 
            preds_dict=preds_dict_unscaled)
    
    print('\n performance metrics logged ')


if __name__ == '__main__':
    evaluate_LSTM()