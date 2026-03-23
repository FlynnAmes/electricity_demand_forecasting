""" script that produces naive forecast for energy demand. 
for now, the script saves the forecast, as well as rmse and nrmse metrics for reporting """

import pandas as pd
from sklearn.metrics import root_mean_squared_error
from paths import LOGS_PATH, DATA_PATH
import json
import os


def create_naive_forecast():
    # load in data
    df = pd.read_csv(DATA_PATH / 'processed' / 'hourly_usage_cleaned.csv')

    # drop the unnamed column (that will be removed later)
    df.drop(columns=['Unnamed: 0'], inplace=True)

    # set index to both datetime and the client id
    df.set_index(['datetime', 'client_id'], inplace=True)

    ####################
    # get predictions
    ####################

    # the first naive prediction is just 24 hours prior lag
    df['yn_1_day_lag'] = df.groupby(level='client_id')['hourly_usage_kwh'].shift(24)

    # then naive prediciton of one week lag
    df['yn_1_wk_lag'] = df.groupby(level='client_id')['hourly_usage_kwh'].shift(7*24)

    # variable for the datetime index for cleaner coding
    datetime_index =  pd.to_datetime(df.index.get_level_values('datetime'))

    # then use mean of that day of week, time and month across all points in dataset
    df['yn_avg_time_day_month'] =  df.groupby([pd.Grouper(level='client_id'),
                                                                    datetime_index.hour,
                                                                    datetime_index.dayofweek,
                                                                    datetime_index.month])['hourly_usage_kwh'].transform('mean')

    # drop nan rows (because data where not possible to predict the target can be omitted)
    df.dropna(inplace=True)


    # get all prediciton columns
    prediction_cols = df.columns[df.columns != 'hourly_usage_kwh']

    # dictionary or rmse values. key is prediction method, values are series containing rmse for each client
    rmse_dict = {col: df.groupby(level='client_id').apply(
        lambda g: root_mean_squared_error(y_true=g['hourly_usage_kwh'], y_pred=g[f'{col}'])) for col in prediction_cols}

    # compute the mean usage for each client
    mean_usage_each_client = df.groupby(level='client_id')['hourly_usage_kwh'].mean()

    # use to normalise the rmse - to get comparison between clients
    nrmse_dict = {col: rmse_dict[f'{col}']/mean_usage_each_client for col in prediction_cols}

    # get the overall mean nrmse for each naive prediction
    nrmse_mean_dict = {col: nrmse_dict[f'{col}'].mean() for col in prediction_cols}

    # convert to lists for serialisation (TODO: figure out cleaner logging system)
    for col in prediction_cols:
        rmse_dict[col] = rmse_dict[col].to_list()
        nrmse_dict[col] = nrmse_dict[col].to_list()

    # save metric to json file (for now removing index, for serialisation)
    os.makedirs(LOGS_PATH / 'naive_forecast', exist_ok=True)

    # rmse
    with open(LOGS_PATH / 'naive_forecast' / 'rmse.json', 'w') as f:
        json.dump(rmse_dict, f, indent=4)

    # rmse normalised by mean usage
    with open(LOGS_PATH / 'naive_forecast' / 'nrmse.json', 'w') as f:
        json.dump(nrmse_dict, f, indent=4)

    # mean of rmse
    with open(LOGS_PATH / 'naive_forecast' / 'nrmse_mean.json', 'w') as f:
        json.dump(nrmse_mean_dict, f, indent=4)

    # and save the corresponding client ids
    with open(LOGS_PATH / 'naive_forecast' / 'client_ids.json', 'w') as f:
        json.dump(df.index.get_level_values('client_id').unique().to_list(), f, indent=4)

    # finally, save the naive forecast itself to a csv file
    df.to_csv(DATA_PATH / 'data' / 'processed' / 'naive_forecasts.csv', index=False)

if __name__ == '__main__':
    create_naive_forecast()


