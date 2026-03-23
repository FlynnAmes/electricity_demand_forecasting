""" code for feature engineering, combined with clean time split of data, to train/test on 
earlier data, and evaluate the model on hold out later bit of data """

import pandas as pd
import numpy as np
import pickle as pkl
import yaml
from paths import DATA_PATH, CONFIG_PATH



def engineer_features():

    # load in config params
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    # get config params
    train_size = config['train_size']

    # load in the data
    df = pd.read_csv(str(DATA_PATH / 'processed' / 'hourly_usage_cleaned.csv'))

    # set datetime and client id as the index, but sort by datetime (for later timeseries split in sklearn)
    df.set_index(['client_id', 'datetime'], inplace=True)
    df.sort_index(level='datetime', inplace=True)


    ##########
    # create new features (grouping by client id!)
    ##########

    # create lagged features
    df['1hr_lag'] = df.groupby(level='client_id')['hourly_usage_kwh'].shift(1)
    df['lag_2hr'] = df.groupby(level='client_id')['hourly_usage_kwh'].shift(2)
    df['lag_6hr'] = df.groupby(level='client_id')['hourly_usage_kwh'].shift(6)
    df['lag_1dy'] = df.groupby(level='client_id')['hourly_usage_kwh'].shift(24)
    df['lag_1wk'] = df.groupby(level='client_id')['hourly_usage_kwh'].shift(7*24)

    # get rolling mean features
    df['1dy_roll_mean'] = df.groupby(level='client_id')['hourly_usage_kwh'].transform(lambda x: x.rolling(window=24).mean())
    df['rolling_mean_1wk'] = df.groupby(level='client_id')['hourly_usage_kwh'].transform(lambda x: x.rolling(window=24*7).mean())

    # get the rolling max, min and standard deviation (for volatility)
    df['rolling_max_1dy'] = df.groupby(level='client_id')['hourly_usage_kwh'].transform(lambda x: x.rolling(window=24).max())
    df['rolling_max_1wk'] = df.groupby(level='client_id')['hourly_usage_kwh'].transform(lambda x: x.rolling(window=24*7).max())

    df['rolling_min_1dy'] = df.groupby(level='client_id')['hourly_usage_kwh'].transform(lambda x: x.rolling(window=24).min())
    df['rolling_min_1wk'] = df.groupby(level='client_id')['hourly_usage_kwh'].transform(lambda x: x.rolling(window=24*7).min())

    df['rolling_std_1dy'] = df.groupby(level='client_id')['hourly_usage_kwh'].transform(lambda x: x.rolling(window=24).std())
    df['rolling_std_1wk'] = df.groupby(level='client_id')['hourly_usage_kwh'].transform(lambda x: x.rolling(window=24*7).std())

    # and now get features for the hour, day of week and month. using cyclical encoding, 
    df['hour'] = np.sin(2*np.pi*pd.to_datetime(df.index.get_level_values('datetime')).hour/24)
    df['day_of_week'] = np.sin(2*np.pi*pd.to_datetime(df.index.get_level_values('datetime')).dayofweek/7)
    df['month'] = np.sin(2*np.pi*pd.to_datetime(df.index.get_level_values('datetime')).month/12)

    # remove nans where no feature
    df.dropna(inplace=True)

    print('\n new features created')

    #############
    # split data in time
    ############

    # first get timestamps for start and end time
    start_date = pd.to_datetime(df.index.get_level_values('datetime')).min()
    end_date = pd.to_datetime(df.index.get_level_values('datetime')).max()

    # then the total time elapsed timedelta
    tot_time_elasped = end_date - start_date

    # multiply by train proportion and add to start date to get cutoff point
    cutoff_date = start_date + tot_time_elasped*train_size

    # now use cutoff data to filter for test and train
    df_train = df[pd.to_datetime(df.index.get_level_values('datetime')) < cutoff_date]
    df_test = df[pd.to_datetime(df.index.get_level_values('datetime')) > cutoff_date]

    print('\n train shape: ', df_train.shape)
    print('\n test shape: ', df_test.shape)

    # now split into target and features
    y_train = df_train['hourly_usage_kwh']
    y_validate = df_test['hourly_usage_kwh']

    X_train = df_train.drop(columns=['hourly_usage_kwh'])
    X_validate = df_test.drop(columns=['hourly_usage_kwh'])

    print('\n data split')

    ###############
    # save processed data
    ###############

    with open(DATA_PATH / 'processed' / 'X_train.pkl', 'wb') as f:
        pkl.dump(X_train, f)

    with open(DATA_PATH / 'processed' /  'y_train.pkl', 'wb') as f:
        pkl.dump(y_train, f)

    with open(DATA_PATH / 'processed' / 'X_validate.pkl', 'wb') as f:
        pkl.dump(X_validate, f)

    with open(DATA_PATH / 'processed' / 'y_validate.pkl', 'wb') as f:
        pkl.dump(y_validate, f)

    print('\n processed data saved')


if __name__ == '__main__':
    engineer_features()





