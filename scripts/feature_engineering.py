""" code for feature engineering, combined with clean time split of data, to train/test on 
earlier data, and evaluate the model on hold out later bit of data """

import pandas as pd
import numpy as np
import pickle as pkl
import yaml
from paths import DATA_PATH, CONFIG_PATH, LOGS_PATH
import json


def engineer_features():

    # load in config params
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    # get config params
    train_size = config['train_size']
    validation_size = config['validation_size']
    test_size = config['test_size']
    num_clients_train = config['num_clients_train']
    random_seed = config['random_seed']

    # load in the data
    df = pd.read_csv(str(DATA_PATH / 'processed' / 'hourly_usage_cleaned.csv'))

    # set datetime and client id as the index, but sort by datetime (for later timeseries split in sklearn)
    df.set_index(['client_id', 'datetime'], inplace=True)

    ##########
    # create new features (for all models)
    ##########

    # extract the datetimes
    datetimes = pd.to_datetime(df.index.get_level_values('datetime'))

    # cyclical time features - used in both tabular and sequence models
    df['hour_sin'] = np.sin(2*np.pi*datetimes.hour/24)
    df['hour_cos'] = np.cos(2*np.pi*datetimes.hour/24)

    # day of week (for diurnal)
    df['day_sin'] = np.sin(2*np.pi*datetimes.dayofweek/7)
    df['day_cos'] = np.cos(2*np.pi*datetimes.dayofweek/7)
    
    # month (for seasonal)
    df['month_sin'] = np.sin(2*np.pi*datetimes.month/12)
    df['month_cos'] = np.cos(2*np.pi*datetimes.month/12)

    print('\n cyclical time features encoded')

    ##############
    # now create additional features (for tabular models will drop these for sequence models)
    ##############

    # create lagged features
    df['lag_1hr'] = df.groupby(level='client_id')['hourly_usage_kwh'].shift(1)
    df['lag_2hr'] = df.groupby(level='client_id')['hourly_usage_kwh'].shift(2)
    df['lag_6hr'] = df.groupby(level='client_id')['hourly_usage_kwh'].shift(6)
    df['lag_1dy'] = df.groupby(level='client_id')['hourly_usage_kwh'].shift(24)
    df['lag_1wk'] = df.groupby(level='client_id')['hourly_usage_kwh'].shift(7*24)

    # get rolling mean features
    df['rolling_mean_1dy'] = df.groupby(level='client_id')['hourly_usage_kwh'].transform(lambda x: x.rolling(window=24).mean())
    df['rolling_mean_1wk'] = df.groupby(level='client_id')['hourly_usage_kwh'].transform(lambda x: x.rolling(window=24*7).mean())

    # get the rolling max, min and standard deviation (for volatility)
    df['rolling_max_1dy'] = df.groupby(level='client_id')['hourly_usage_kwh'].transform(lambda x: x.rolling(window=24).max())
    df['rolling_max_1wk'] = df.groupby(level='client_id')['hourly_usage_kwh'].transform(lambda x: x.rolling(window=24*7).max())

    df['rolling_min_1dy'] = df.groupby(level='client_id')['hourly_usage_kwh'].transform(lambda x: x.rolling(window=24).min())
    df['rolling_min_1wk'] = df.groupby(level='client_id')['hourly_usage_kwh'].transform(lambda x: x.rolling(window=24*7).min())

    df['rolling_std_1dy'] = df.groupby(level='client_id')['hourly_usage_kwh'].transform(lambda x: x.rolling(window=24).std())
    df['rolling_std_1wk'] = df.groupby(level='client_id')['hourly_usage_kwh'].transform(lambda x: x.rolling(window=24*7).std())

    print('\n new features created')

    #############
    # get cutoff dates for time split
    ############

    # split data into train, validation and holdout test, with 70, 10, 20 split
    min_date = datetimes.min()
    max_date = datetimes.max()

    # cutoff date where validation data begins. NOTE that proportion in training will be 
    # slihgtly smaller than specified proportion because many clients don't have data for first year
    validation_cutoff = min_date + train_size*(max_date - min_date)
    # cutoff date where test data begins
    test_cutoff = min_date + (train_size + validation_size)*(max_date - min_date)

    # split df into train, validate and test
    df_train = df[datetimes < validation_cutoff]
    df_validate = df[(datetimes > validation_cutoff) & (datetimes < test_cutoff)]
    df_test = df[datetimes > test_cutoff]

    print('\n shape of train: ', df_train.shape)
    print('shape of validate: ', df_validate.shape)
    print('shape of test: ', df_test.shape)

    ##########
    # extract random subset of clients to manage compute
    #########

    # get client ids in all rows in train, validation and test periods
    client_ids_train = df_train.index.get_level_values('client_id')
    client_ids_validate = df_validate.index.get_level_values('client_id')
    client_ids_test = df_test.index.get_level_values('client_id')

    # create a list of client ids that are in all training, testing and validation (train misses some that are in validate/test)
    client_ids_in_all_data = client_ids_train.intersection(client_ids_validate).intersection(client_ids_test)

    # create random number generator with seed
    rng = np.random.default_rng(seed=random_seed)
    random_client_subset = np.sort(rng.choice(client_ids_in_all_data, size=num_clients_train, replace=False))

    # filter so only including random client subset
    df_train_filtered = df_train[client_ids_train.isin(random_client_subset)]
    df_validate_filtered = df_validate[client_ids_validate.isin(random_client_subset)]
    df_test_filtered = df_test[client_ids_test.isin(random_client_subset)]

    print('\n shape of train after filter: ', df_train_filtered.shape)
    print('shape of validate after filter: ', df_validate_filtered.shape)
    print('shape of test after filter: ', df_test_filtered.shape)

    print('\n random subset of clients extracted')

    # log subset of clients used
    with open(LOGS_PATH / 'model_development' / 'clients_used_training.json', 'w') as f:
        json.dump(list(df_train_filtered.index.get_level_values('client_id').unique()), f , indent=4)

    ################
    # now drop lagged/rolled feature columns for sequential data
    ################

    lag_roll_feat_cols = df.columns.difference(set(('hourly_usage_kwh', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos')))
    
    df_seq_train = df_train_filtered.drop(columns=lag_roll_feat_cols)
    df_seq_validate = df_validate_filtered.drop(columns=lag_roll_feat_cols)
    df_seq_test = df_test_filtered.drop(columns=lag_roll_feat_cols)

    print('\n cols in sequential: ', df_seq_train.columns)

    #  # and save the sequential models
    # df_seq_train.to_csv(DATA_PATH / 'processed' / 'df_seq_train.csv')
    # df_seq_validate.to_csv(DATA_PATH / 'processed' / 'df_seq_validate.csv')
    # df_seq_test.to_csv(DATA_PATH / 'processed' / 'df_seq_test.csv')

    # and save the sequential models
    with open(DATA_PATH / 'processed' / 'df_seq_train.pkl', 'wb') as f:
        pkl.dump(df_seq_train, f)

    with open(DATA_PATH / 'processed' / 'df_seq_validate.pkl', 'wb') as f:
        pkl.dump(df_seq_validate, f)
     
    with open(DATA_PATH / 'processed' / 'df_seq_test.pkl', 'wb') as f:
        pkl.dump(df_seq_test, f)

    print('\n engineered sequential model data saved')

    # for tabular models, first drop columns where rolled/lag cannot be computed. 
    df_train_filtered.dropna(inplace=True)
    df_validate_filtered.dropna(inplace=True)
    df_test_filtered.dropna(inplace=True)

    # # and then save for tabular data
    # df_train_filtered.to_csv(DATA_PATH / 'processed' / 'df_tabular_train.csv')
    # df_validate_filtered.to_csv(DATA_PATH / 'processed' / 'df_tabular_validate.csv')
    # df_test_filtered.to_csv(DATA_PATH / 'processed' / 'df_tabular_test.csv')

    with open(DATA_PATH / 'processed' / 'df_tabular_train.pkl', 'wb') as f:
        pkl.dump(df_train_filtered, f)

    with open(DATA_PATH / 'processed' / 'df_tabular_validate.pkl', 'wb') as f:
        pkl.dump(df_validate_filtered, f)
    
    with open(DATA_PATH / 'processed' / 'df_tabular_test.pkl', 'wb') as f:
        pkl.dump(df_test_filtered, f)

    print('\n engineered data for tabular models saved')


if __name__ == '__main__':
    engineer_features()





