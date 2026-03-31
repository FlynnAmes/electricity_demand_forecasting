""" Train LSTM model """

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import pickle as pkl
import numpy as np
import yaml
from paths import CONFIG_PATH, DATA_PATH, BASE_PATH, MODELS_PATH
from classes import SeqExtractionDataSet, LSTMNoEmbed, ConvergenceTracker


##############
# load params from config
##############

def train_LSTM():

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
        LSTM_params = config['LSTM_params']

    SEQ_LENGTH = LSTM_params['seq_length']  # length of context window
    HIDDEN_SIZE = LSTM_params['hidden_size'] # hidden size in LSTM

    LEARNING_RATE = LSTM_params['learning_rate']  # base learning rate in optimiser
    WEIGHT_DECAY = LSTM_params['weight_decay']  # weight decay in optimiser

    EPOCHS = LSTM_params['epochs']  # epochs to run through data during training
    BATCH_SIZE = LSTM_params['batch_size']  # size of batches used during training
    PATIENCE = LSTM_params['patience']  # patience and delta for early stopping
    DELTA = LSTM_params['delta']

    #######################
    # load dictionaries containing data for each client, as well as index maps
    #######################

    # data dictionaries
    with open(DATA_PATH / 'processed' / 'seq_data_dict_train', 'rb') as f:
        data_dict_train = pkl.load(f)

    with open(DATA_PATH / 'processed' / 'seq_data_dict_validate', 'rb') as f:
        data_dict_validate = pkl.load(f)

    # index maps
    with open(DATA_PATH / 'processed' / 'idx_map_train', 'rb') as f:
        idx_map_train = pkl.load(f)

    with open(DATA_PATH / 'processed' / 'idx_map_validate', 'rb') as f:
        idx_map_validate = pkl.load(f)

    ##################
    # set up datasets and dataloaders
    ##################

    # now create instance of the dataset for train, test, and validate
    dataset_train = SeqExtractionDataSet(data_dict=data_dict_train, idx_map=idx_map_train, SEQ_LENGTH=SEQ_LENGTH)
    dataset_validate = SeqExtractionDataSet(data_dict=data_dict_validate, idx_map=idx_map_validate, SEQ_LENGTH=SEQ_LENGTH)

    # and corresponding dataloaders. Fine to shuffle because using sequences here rather than time points
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    dataloader_validate = DataLoader(dataset_validate, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    #########
    # Set up model, optimiser, and loss function
    #########

    # get number of columns from training data (using data from data dict)
    n_features = data_dict_train[idx_map_train[0][0]].shape[1]

    # set up model instance
    model = LSTMNoEmbed(n_features=n_features, hidden_size=HIDDEN_SIZE)

    # AdamW with more consistent regularisation accross weights
    optimiser = AdamW(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # use MSE for criterion for training
    criterion = nn.MSELoss()

    # create instance of early stopping class
    convergence_tracker = ConvergenceTracker(patience=PATIENCE, delta=DELTA)

    ###########
    # main training loop
    ###########
    try:
        # first loop over epochs
        for epoch in range(EPOCHS):

            print(f'\n starting epoch {epoch}')
            losses_train = []
            model.train()
            
            for batch_idx, (X_seqs, labels, _) in enumerate(dataloader_train):

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
                # for every 5th epoch, plot losses
                convergence_tracker.plot_losses(show_figure=False, save_figure=True, 
                                                save_path=BASE_PATH / 'plots')
                # and save model checkpoint
                with open(MODELS_PATH / 'checkpoints' / 'LSTM_checkpoint.pkl', 'wb') as f:
                    pkl.dump(model.state_dict(), f)
                

    # catch keyboard interupt
    except KeyboardInterrupt:
        print('training interupted by user')
    
    # catch any other error
    except Exception as e:
        print(f'training failed with error: \n {e}')
        raise
    
    # always save a model checkpoint erroring or not
    finally:
        with open(MODELS_PATH / 'checkpoints' / 'LSTM_checkpoint.pkl', 'wb') as f:
            pkl.dump(model.state_dict(), f)
            print('model checkpoint saved')


if __name__ == '__main__':
    train_LSTM()