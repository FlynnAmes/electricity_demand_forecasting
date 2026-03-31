""" prepare data for sequential models, creating dictionaries with data per client, along with 
index map giving all possible combinations of client id and starting index for a given sequence length """

import torch
import pickle as pkl
import numpy as np
import yaml
from paths import CONFIG_PATH, DATA_PATH


with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)
    LSTM_params = config['LSTM_params']

# length of context window for prediction
SEQ_LENGTH = LSTM_params['seq_length'] 
# min and max size of random 'strides' used during training to obtain random subsample of time points
TRAIN_STRIDE_LOW = LSTM_params['train_stride_low']  
TRAIN_STRIDE_HIGH = LSTM_params['train_stride_high']
# fixed stride for validation and testing to ensure reproducibility. Latter is 1 to siulate production.
VALIDATE_STRIDE = LSTM_params['validate_stride'] 
TEST_STRIDE = LSTM_params['test_stride']
# random seed for sampling of stride for training
RANDOM_SEED = config['random_seed']

#######################
# load and ready the training and validation data
#######################

for d in ['train', 'validate', 'test']:

    # loading in the training, validation and testing data
    with open(DATA_PATH / 'processed' / f'df_seq_{d}.pkl', 'rb') as f:
        # load in dataframe
        df = pkl.load(f)
    
    # convert to tensor ready for PyTorch
    data = torch.tensor(df.to_numpy(), dtype=torch.float32)

    ###################
    # Create data dicts and index maps
    ###################

    # get client ids
    client_ids = df.index.get_level_values('client_id')
    client_ids_unique = client_ids.unique()

    # create dictionary with client id as key and corresponding data (tensor) as value
    data_dict = {client_id: data[client_ids == client_id]
                    for client_id in client_ids_unique}
    
    ###############
    # and index maps for extracting data from dictionary
    ###############

    # random number generator for stide in training sequences
    rng = np.random.default_rng(seed=RANDOM_SEED)

    # total valid number starting indexes (and thus valid number of sequences) for each client
    tot_num_start_idx = [data.shape[0] - SEQ_LENGTH for client_id, data in data_dict.items()]
    
    if d == 'train':
        # for training data, use random stride to subsample data
        index_map = [(client_id, start_index) for i, client_id in enumerate(client_ids_unique)
                            for start_index in range(0, tot_num_start_idx[i], rng.integers(low=TRAIN_STRIDE_LOW, high=TRAIN_STRIDE_HIGH, size=1)[0])]
    elif d == 'validate':
        # validation uses fixed stride
        index_map = [(client_id, start_index) for i, client_id in enumerate(client_ids_unique)
                            for start_index in range(0, tot_num_start_idx[i], VALIDATE_STRIDE)]
    elif d == 'test':
        # same for testing 
        index_map = [(client_id, start_index) for i, client_id in enumerate(client_ids_unique)
                            for start_index in range(0, tot_num_start_idx[i], TEST_STRIDE)]
    else:
        raise NameError(f'name of data to preprocess should be train, test or validate, currently {d}') 

    # now save the index maps
    with open(DATA_PATH / 'processed' / f'idx_map_{d}', 'wb') as f:
        pkl.dump(index_map, f)

    # and dictionaries containing the data per client
    with open(DATA_PATH / 'processed' / f'seq_data_dict_{d}', 'wb') as f:
        pkl.dump(data_dict, f)

    print(f'\n {d} sequential data preprocessed')