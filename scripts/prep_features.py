""" code for feature engineering, combined with clean time split of data, to train/test on 
earlier data, and evaluate the model on hold out later bit of data """

import pandas as pd
import numpy as np
import pickle as pkl
import yaml
from paths import DATA_PATH, CONFIG_PATH, LOGS_PATH
import json



def standardise_per_client(group, dict_with_mean_and_std):
    """ standardise hourly usage data using mean and standard deviation for each client """

    # get client id of a group (assumed the same within group)
    client_id = int(np.unique(group.index.get_level_values('client_id'))[0])
    # for each column in the group, update values using corresponding mean and std
    return (group - dict_with_mean_and_std[client_id]['mean'])/dict_with_mean_and_std[client_id]['std']


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


    #############
    # get cutoff dates for time split
    ############
    
    # extract the datetimes
    datetimes = pd.to_datetime(df.index.get_level_values('datetime'))

    # split data into train, validation and holdout test, using split from config
    min_date = datetimes.min()
    max_date = datetimes.max()

    # cutoff date where validation data begins. NOTE that proportion in training will be 
    # slightly smaller than specified proportion because many clients don't have data for first year
    validation_cutoff = min_date + train_size*(max_date - min_date)
    # cutoff date where test data begins
    test_cutoff = min_date + (train_size + validation_size)*(max_date - min_date)

    ###############
    # random subset of clients to manage compute
    ###############

    # get client ids in all rows in train, validation and test periods
    client_ids_train = df[datetimes < validation_cutoff].index.get_level_values('client_id')
    client_ids_validate = df[(datetimes > validation_cutoff) & (datetimes < test_cutoff)].index.get_level_values('client_id')
    client_ids_test = df[datetimes > test_cutoff].index.get_level_values('client_id')

    # create a list of client ids that are in all training, testing and validation (train misses some that are in validate/test)
    client_ids_in_all_data = client_ids_train.intersection(client_ids_validate).intersection(client_ids_test)

    # create random number generator with seed
    rng = np.random.default_rng(seed=random_seed)
    random_client_subset = np.sort(rng.choice(client_ids_in_all_data, size=num_clients_train, replace=False))

    # now filter main df, so that only include clients from this subset
    df_filtered = df[df.index.get_level_values('client_id').isin(random_client_subset)]
    
    print('\n orig shape: ', df.shape)
    print('\n filtered shape: ', df_filtered.shape)

    #################
    # standardise the hourly usage on a per client basis (using training data)
    ################

    # update to reflect datetimes in data subset
    datetimes = pd.to_datetime(df_filtered.index.get_level_values('datetime'))
  
    # get mean and std hourly usage for each client using TRAINING DATA ONLY
    df_mean_std_usage = df_filtered[datetimes < validation_cutoff]['hourly_usage_kwh'].groupby(level='client_id').agg(['mean', 'std'])

    # convert to a dictionary for accessing later. Use index as key
    dict_mean_std_usage = df_mean_std_usage.to_dict(orient='index')

    # now standardise ALL data using client-specific mean and std from TRAINING 
    df_standardised = df_filtered.groupby(level='client_id').transform(lambda g: standardise_per_client(g, dict_mean_std_usage))

    ##########
    # create new time features (for all models)
    ##########

    # cyclical time features - used in both tabular and sequence models
    df_standardised['hour_sin'] = np.sin(2*np.pi*datetimes.hour/24)
    df_standardised['hour_cos'] = np.cos(2*np.pi*datetimes.hour/24)

    # day of week (for diurnal)
    df_standardised['day_sin'] = np.sin(2*np.pi*datetimes.dayofweek/7)
    df_standardised['day_cos'] = np.cos(2*np.pi*datetimes.dayofweek/7)
    
    # month (for seasonal)
    df_standardised['month_sin'] = np.sin(2*np.pi*datetimes.month/12)
    df_standardised['month_cos'] = np.cos(2*np.pi*datetimes.month/12)

    print('\n cyclical time features encoded')

    ##############
    # now create additional features (for tabular models will drop these for sequence models)
    ##############

    # create lagged features
    df_standardised['lag_1hr'] = df_standardised.groupby(level='client_id')['hourly_usage_kwh'].shift(1)
    df_standardised['lag_2hr'] = df_standardised.groupby(level='client_id')['hourly_usage_kwh'].shift(2)
    df_standardised['lag_6hr'] = df_standardised.groupby(level='client_id')['hourly_usage_kwh'].shift(6)
    df_standardised['lag_1dy'] = df_standardised.groupby(level='client_id')['hourly_usage_kwh'].shift(24)
    df_standardised['lag_1wk'] = df_standardised.groupby(level='client_id')['hourly_usage_kwh'].shift(7*24)

    # get rolling mean features
    df_standardised['rolling_mean_1dy'] = df_standardised.groupby(level='client_id')['hourly_usage_kwh'].transform(lambda x: x.rolling(window=24).mean())
    df_standardised['rolling_mean_1wk'] = df_standardised.groupby(level='client_id')['hourly_usage_kwh'].transform(lambda x: x.rolling(window=24*7).mean())

    # get the rolling max, min and standard deviation (for volatility)
    df_standardised['rolling_max_1dy'] = df_standardised.groupby(level='client_id')['hourly_usage_kwh'].transform(lambda x: x.rolling(window=24).max())
    df_standardised['rolling_max_1wk'] = df_standardised.groupby(level='client_id')['hourly_usage_kwh'].transform(lambda x: x.rolling(window=24*7).max())

    df_standardised['rolling_min_1dy'] = df_standardised.groupby(level='client_id')['hourly_usage_kwh'].transform(lambda x: x.rolling(window=24).min())
    df_standardised['rolling_min_1wk'] = df_standardised.groupby(level='client_id')['hourly_usage_kwh'].transform(lambda x: x.rolling(window=24*7).min())

    df_standardised['rolling_std_1dy'] = df_standardised.groupby(level='client_id')['hourly_usage_kwh'].transform(lambda x: x.rolling(window=24).std())
    df_standardised['rolling_std_1wk'] = df_standardised.groupby(level='client_id')['hourly_usage_kwh'].transform(lambda x: x.rolling(window=24*7).std())

    print('\n new features created')

    ##############
    # get mean usage per client
    ##############

    # get mean and standard deviations of mean-usage across clients
    mean_of_mean_usages = df_mean_std_usage['mean'].mean()
    std_of_mean_usages = df_mean_std_usage['mean'].std()

    # use to standardise mean usages (using training data only!)
    scaled_mean_client_usage = (df_mean_std_usage['mean'] - mean_of_mean_usages)/std_of_mean_usages

 
    # make this an additional feature to the df. reste and reassign index so that maintain datetime there
    df_standardised = pd.merge(df_standardised.reset_index(), scaled_mean_client_usage, how='left', on=['client_id'])
    # rename merged column
    df_standardised.rename(columns={'mean': 'mean_usage'}, inplace=True)
    # and reassign client id and datetime to index
    df_standardised.set_index(['client_id', 'datetime'], inplace=True)


    #####################
    # split into train, validate, test and SAVE
    ######################

    # train validation test split for tabular data
    df_train = df_standardised[datetimes < validation_cutoff]
    df_validate = df_standardised[(datetimes > validation_cutoff) & (datetimes < test_cutoff)]
    df_test = df_standardised[datetimes > test_cutoff]

    # drop na values from each
    df_train.dropna(inplace=True)
    df_validate.dropna(inplace=True)
    df_test.dropna(inplace=True)

    # and save this data (for use in classical models)
    with open(DATA_PATH / 'processed' / 'df_tabular_train.pkl', 'wb') as f:
        pkl.dump(df_train, f)

    with open(DATA_PATH / 'processed' / 'df_tabular_validate.pkl', 'wb') as f:
        pkl.dump(df_validate, f)
    
    with open(DATA_PATH / 'processed' / 'df_tabular_test.pkl', 'wb') as f:
        pkl.dump(df_test, f)

    print('\n engineered data for tabular models saved')

    ################
    # now drop lagged/rolled feature columns for sequential data and save this version
    ################

    lag_roll_feat_cols = df_standardised.columns.difference(set(('hourly_usage_kwh', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 'mean_usage')))
    df_train.drop(columns=lag_roll_feat_cols, inplace=True)
    df_validate.drop(columns=lag_roll_feat_cols, inplace=True)
    df_test.drop(columns=lag_roll_feat_cols, inplace=True)

    # and save the data for sequential models
    with open(DATA_PATH / 'processed' / 'df_seq_train.pkl', 'wb') as f:
        pkl.dump(df_train, f)

    with open(DATA_PATH / 'processed' / 'df_seq_validate.pkl', 'wb') as f:
        pkl.dump(df_validate, f)
     
    with open(DATA_PATH / 'processed' / 'df_seq_test.pkl', 'wb') as f:
        pkl.dump(df_test, f)

    print('\n engineered sequential model data saved')


    # save the dictionary containing mean and standard deviation usage for each client
    with open(DATA_PATH / 'processed' / 'mean_std_per_client.json', 'w') as f:
        json.dump(dict_mean_std_usage, f)


if __name__ == '__main__':
    engineer_features()





