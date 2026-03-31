""" Train LSTM model """

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import Dataset, DataLoader
import pickle as pkl
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import json
import yaml
from paths import CONFIG_PATH, DATA_PATH, BASE_PATH, MODELS_PATH
from classes import SeqExtractionDataSet, LSTMNoEmbed, ConvergenceTracker


##############
# load params from config
##############

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)
    LSTM_params = config['LSTM_params']

RANDOM_SEED = config['random_seed']
SEQ_LENGTH = LSTM_params['seq_length']  # length of context window for prediction

TRAIN_STRIDE_LOW = LSTM_params['train_stride_low']  # size of random 'strides' used during training to obtain random subsample of time points
TRAIN_STRIDE_HIGH = LSTM_params['train_stride_high']
VALIDATE_STRIDE = LSTM_params['validate_stride']  # fixed stride for validation to ensure reproducibility
TEST_STRIDE = LSTM_params['test_stride']  # stride of 1 for testing to simulate production

HIDDEN_SIZE = LSTM_params['hidden_size'] # hidden size in LSTM

LEARNING_RATE = LSTM_params['learning_rate']  # base learning rate in optimiser
WEIGHT_DECAY = LSTM_params['weight_decay']  # weight decay in optimiser

EPOCHS = LSTM_params['epochs']  # epochs to run through data during training
BATCH_SIZE = LSTM_params['batch_size']  # size of batches used during training
PATIENCE = LSTM_params['patience']  # patience and delta for early stopping
DELTA = LSTM_params['delta']

#######################
# load and ready the training and validation data
#######################

# loading in the training, validation and testing data
with open(DATA_PATH / 'processed' / 'df_seq_train.pkl', 'rb') as f:
    df_train = pkl.load(f)

with open(DATA_PATH / 'processed' / 'df_seq_validate.pkl', 'rb') as f:
    df_validate = pkl.load(f)

with open(DATA_PATH / 'processed' / 'df_seq_test.pkl', 'rb') as f:
    df_test = pkl.load(f)

# convert to tensor format ready for PyTorch
data_train_preprocesed = torch.tensor(df_train.to_numpy(), dtype=torch.float32)
data_validate_preprocesed = torch.tensor(df_validate.to_numpy(), dtype=torch.float32)
data_test_preprocesed = torch.tensor(df_test.to_numpy(), dtype=torch.float32)

###################
# Create data dicts and index maps
###################

# get client ids (for readability)
client_ids_train = df_train.index.get_level_values('client_id')
client_ids_validate = df_validate.index.get_level_values('client_id')
client_ids_test = df_test.index.get_level_values('client_id')

# unique client ids in data (ensured during feature engineering that same across all data)
client_ids_unique = client_ids_validate.unique()

# create dictionaries of data where key represents client and value all data for that client
# (that will access during training)
data_train_dict = {client_id: data_train_preprocesed[client_ids_train == client_id]
                    for client_id in client_ids_unique}

data_validate_dict = {client_id: data_validate_preprocesed[client_ids_validate == client_id]
                       for client_id in client_ids_unique}

data_test_dict = {client_id: data_test_preprocesed[client_ids_test == client_id]
                   for client_id in client_ids_unique}

# total valid number starting indexes (and thus valid number of sequences) for each client
tot_num_start_indx_train = [data.shape[0] - SEQ_LENGTH for client_id, data in data_train_dict.items()]
tot_num_start_indx_validate = [data.shape[0] - SEQ_LENGTH for client_id, data in data_validate_dict.items()]
tot_num_start_indx_test = [data.shape[0] - SEQ_LENGTH for client_id, data in data_test_dict.items()]

# random number generator for stide in training sequences
rng = np.random.default_rng(seed=RANDOM_SEED)

# random goes isnide loop here so that random stride at every iteration
index_map_train = [(client_id, start_index) for i, client_id in enumerate(client_ids_unique)
                    for start_index in range(0, tot_num_start_indx_train[i], rng.integers(low=TRAIN_STRIDE_LOW, high=TRAIN_STRIDE_HIGH, size=1)[0])]

index_map_validate = [(client_id, start_index) for i, client_id in enumerate(client_ids_unique)
                       for start_index in range(0, tot_num_start_indx_validate[i], VALIDATE_STRIDE)]

index_map_test = [(client_id, start_index) for i, client_id in enumerate(client_ids_unique)
                   for start_index in range(0, tot_num_start_indx_test[i], TEST_STRIDE)]

##################
# set up datasets and dataloaders
##################

# now create instance of the dataset for train, test, and validate
dataset_train = SeqExtractionDataSet(data_dict=data_train_dict, idx_map=index_map_train, SEQ_LENGTH=SEQ_LENGTH)
dataset_validate = SeqExtractionDataSet(data_dict=data_validate_dict, idx_map=index_map_validate, SEQ_LENGTH=SEQ_LENGTH)
dataset_test = SeqExtractionDataSet(data_dict=data_test_dict, idx_map=index_map_test, SEQ_LENGTH=SEQ_LENGTH)

# and corresponding dataloaders. Fine to shuffle because using sequences here rather than time points
dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
dataloader_validate = DataLoader(dataset_validate, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
# don't bother shuffling test as not computing the loss here (and don't drop last batch as will concat)
dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

#########
# Set up model, optimiser, and loss function
#########

# get number of columns from training data
n_features = len(df_train.columns)

# set up model instance
model = LSTMNoEmbed(n_features=n_features, hidden_size=HIDDEN_SIZE)

#  AdamW with more consistent regularisation accross weights
optimiser = AdamW(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# use MSE for criterion for training
criterion = nn.MSELoss()

# create instance of early stopping class
convergence_tracker = ConvergenceTracker(patience=PATIENCE, delta=DELTA)

###########
# main training loop
###########

# first loop over epochs
for epoch in range(EPOCHS):

    print(f'\n starting epoch {epoch}')
    losses_train = []
    model.train()
    
    for batch_idx, (X_seqs, labels, client_ids) in enumerate(dataloader_train):

        # do forwards pass
        predictions = model(X_seqs)

        # compute loss
        loss = criterion(predictions, labels)
        loss.backward()

        # step forwards
        optimiser.step()
        optimiser.zero_grad()
    
        # append batch training loss to list
        with torch.no_grad():
            losses_train.append(loss.item())

    print(f'training at epoch {epoch} completed')
    
    with torch.no_grad():
        # after training completed, append batch-averaged training loss to list
        convergence_tracker.training_losses.append(np.mean(losses_train))

    # at every other epoch
    if epoch % 2 == 0:
        
        # list of losses which will average over (reset at each validation loop)
        losses_val = []
        # at end of epoch, now evaluate the validation loss
        model.eval()
        with torch.no_grad():
            for batch_idx, (X_seqs_val, labels_val, _) in enumerate(dataloader_validate):

                preds_val = model(X_seqs_val)
                losses_val.append(criterion(preds_val, labels_val).item())
            
            # after going through batches average loss across them
            losses_val_average = np.mean(losses_val)

            # and evaluate early stopping
            convergence_tracker.check_early_stop(val_loss=losses_val_average, epoch=epoch, model_state_dict=model.state_dict())

            # if time to stop then do so and restore best performing weights and biases
            if convergence_tracker.stop_training is True:
                print(f'early stopping criterion reached at {epoch} epochs. restoring to weights and biases at {convergence_tracker.prev_dec_epoch} epochs')

                # model.load_state_dict(early_stop_checker.best_model)
                model.load_state_dict(convergence_tracker.prev_dec_model)

                # and save the mode state dictionary
                with open(MODELS_PATH / 'LSTM.pkl', 'wb') as f:
                    pkl.dump(convergence_tracker.prev_dec_model, f)
                
                # then end training
                break
                
            # append average loss to its own list for plotting later
            convergence_tracker.validation_losses.append(losses_val_average)

    if epoch % 5 == 0:
        convergence_tracker.plot_losses(show_figure=False, save_figure=True, 
                                        save_path=BASE_PATH / 'plots')