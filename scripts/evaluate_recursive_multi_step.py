""" evaluate models doing multi-timestep forecast (using predictions to recursively forecast future time 
    points) """

import pandas as pd
import pickle as pkl
from sklearn.metrics import root_mean_squared_error
import numpy as np
import json
from paths import DATA_PATH, MODELS_PATH, LOGS_PATH, CONFIG_PATH
from glob import glob
from pathlib import Path
import os
import yaml


def unscale_per_client(group, df_with_mean_std_usage):
    """ use mean and std usage for each client to unscale data, ready for 
    evaluation """

    client_id = int(np.unique(group.index.get_level_values('client_id'))[0])

    return ((group * df_with_mean_std_usage.loc[client_id, 'std_usage']) + df_with_mean_std_usage.loc[client_id, 'mean_usage'])


def get_test_data():

    """ return the feature and target test data """

    df = pd.read_parquet(DATA_PATH / 'processed' / 'df_tabular_test.parquet').set_index(['client_id', 'target_time'])

    X_test = df.drop(columns=['target_hourly_usage'])
    y_test = df['target_hourly_usage']

    return X_test, y_test


def log_stats(model_name: str, nrmse_per_client, nrmse_summary_dict, df_preds):
    """ log performance metrics acrosss clients along with summary stats of these, and 
    the predictions for future visualisation """

    # define path to save and check that exists
    save_path = LOGS_PATH / model_name / 'recursive'
    os.makedirs(save_path, exist_ok=True)

    with open(save_path / 'nrmse_summary_stats.json', 'w') as f:
        json.dump(nrmse_summary_dict, f, indent=4)

    with open(save_path / 'nrmse_per_client.json', 'w') as f:
        # convert to list so serialisable
        json.dump(nrmse_per_client.to_list(), f, indent=4)
    
    # save predictions (only) for further evaluation/visualisation (also gives client ids used)
    os.makedirs(DATA_PATH / 'processed', exist_ok=True)
    with open(DATA_PATH / 'processed' / f'{model_name}_preds_multistep.pkl', 'wb') as f:
        pkl.dump(df_preds, f)



def remove_first_week_and_up_to_first_6am_before_last_5am(g):
    """ takes a group and removes the first week of data (naive assumption that all data included for now!)
    and then data up to the first 6 am (first forecast step) """

    # first remove first week
    g_minus_1wk = g[168:]

    # get min datetime that is 6am
    datetimes = g_minus_1wk.index.get_level_values('target_time')
    # client_id = g.index.get_level_values('client_id').unique()
    min_6am = np.min(datetimes[datetimes.hour == 6])
    max_6am = np.max(datetimes[datetimes.hour == 6])

    # get index of min 6am and max 5am
    idx_min_6am = g_minus_1wk.index.droplevel('client_id').get_loc(min_6am)
    idx_max_6am = g_minus_1wk.index.droplevel('client_id').get_loc(max_6am)

    if np.isnan(idx_max_6am).any() or np.isnan(idx_min_6am).any():
        raise Exception(f'idx max is {idx_max_6am} and min {idx_min_6am} for client {g.index.get_level_values('client_id').unique()}')

    return g.iloc[idx_min_6am:idx_max_6am]



def update_the_feature_array(initial_features_indexes, hour_index, col_idxs, X_test_array, y_test_array, y_preds_array, context_window=168):

    """ take the index of data to collect and recompute lag and rolling features, ready for 
    prediction (default context window of 168 steps (assuming 1 week for contigous data) """
    
    # get exisiting feature values at each initial forecast step (using single step
    # engineered features as baseline) 
    feature_array = X_test_array[initial_features_indexes + hour_index, :]
    
    # expand dims for indexes so can broadcast later when want range of indexes
    initial_features_indexes_2d = np.expand_dims(initial_features_indexes, axis=-1)

    # use np.arange to extract contigous indexes (i.e., window; first axis) for each context window (zeroth axis)
    data_from_target = y_test_array[initial_features_indexes_2d + hour_index - context_window + np.arange(context_window - hour_index)]
    # then get data from preds
    data_from_preds = y_preds_array[:, :hour_index]
    # then combine
    data_to_process = np.concatenate((data_from_target, data_from_preds), axis=-1)

    # recompute lag features
    feature_array[:, col_idxs['lag_1hr']] = data_to_process[:, -1]
    feature_array[:, col_idxs['lag_2hr']] = data_to_process[:, -2]
    feature_array[:, col_idxs['lag_6hr']] = data_to_process[:, -6]
    feature_array[:, col_idxs['lag_1dy']] = data_to_process[:, -24]
    feature_array[:, col_idxs['lag_1wk']] = data_to_process[:, -168]

    # and the rolling features
    feature_array[:, col_idxs['rolling_mean_1dy']] = np.mean(data_to_process[:, -24:], axis=-1)
    feature_array[:, col_idxs['rolling_mean_1wk']] = np.mean(data_to_process[:, -168:], axis=-1)

    feature_array[:, col_idxs['rolling_max_1dy']] = np.max(data_to_process[:, -24:], axis=-1)
    feature_array[:, col_idxs['rolling_max_1wk']] = np.max(data_to_process[:, -168:], axis=-1)

    feature_array[:, col_idxs['rolling_min_1dy']] = np.min(data_to_process[:, -24:], axis=-1)
    feature_array[:, col_idxs['rolling_min_1wk']] = np.min(data_to_process[:, -168:], axis=-1)

    return feature_array


def get_forecast(model_object, X_test_array, y_test_array, initial_features_indexes, forecast_horizon, col_idxs, context_window=168):
    """ step forwards the forecast in time to get predictions at each hour of day """

    # initialise array for predictions (num_forecasts, hour_of_day)
    y_preds_array = np.zeros((len(initial_features_indexes), forecast_horizon))

    # loop over each hour to forecast (index zero starts at 6am)
    for hour_index in range(forecast_horizon):

        if hour_index == 0:
            # if hour is zero, then just take the feature values from test data (already in correct order)
            feature_array = X_test_array[initial_features_indexes]
        else:
            # otherwise concat exisiting observed data with predictions (recursive step)
            feature_array = update_the_feature_array(initial_features_indexes=initial_features_indexes, 
                                                    hour_index=hour_index, X_test_array=X_test_array, y_test_array=y_test_array, 
                                                    col_idxs=col_idxs, y_preds_array=y_preds_array,
                                                    context_window=context_window)
            
            if np.isnan(feature_array).any():
                raise Exception('feature array contains nans')
        
        # update the predictions matrix
        y_preds_array[:, hour_index] = model_object.predict(feature_array)

        if np.isnan(y_preds_array).any():
                raise Exception('feature array contains nans')
    
    return y_preds_array


def evaluate_models():

    ###########
    # get config params
    ###########

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    HORIZON = config['horizon']
    CONTEXT_WINDOW = config['context_window']

    ############
    # load test data
    ############

    X_test, y_test = get_test_data()

    print('\n data loaded')

    # test that all groups sorted by datetime in ascending order (neccesary for indexing into datetimes) otherwise raise exception
    if ~X_test.groupby(level='client_id').apply(lambda g: g.index.get_level_values('target_time').is_monotonic_increasing).any():
        raise Exception('datetimes are not montonic increasing for at least one client')
        
    ############
    # load in mean and std usages for each client.
    ############

    df_mean_std_usages = pd.read_json(DATA_PATH / 'processed' / 'mean_std_usages_per_client.json', lines=True).set_index('client_id')
    # get clients used in data
    clients_in_data = pd.read_json(DATA_PATH / 'processed' / 'client_subset_for_training.json', lines=True)
    # ensure only left with mean and std usages used in data
    df_mean_std_usages_for_data = df_mean_std_usages[df_mean_std_usages.index.isin(clients_in_data.to_numpy().squeeze())]
    
    ##########
    # get indexes of predictions and columns, and convert to numpy
    ##########

    # get indexes for all timesteps (and correspinding client id) where a forecast prediction is to be made
    indexes_of_all_predictions = X_test.groupby(level='client_id').apply(lambda g: remove_first_week_and_up_to_first_6am_before_last_5am(g)).index.droplevel(0)

    # get datetimes (and corresponding client id) for starting step of each forecast horizon (i.e., 6am)
    datetimes_forecast_start = indexes_of_all_predictions[indexes_of_all_predictions.get_level_values('target_time').hour == 6]

    # get the numeric index for starting step of each forecast
    initial_features_indexes = X_test.index.get_indexer(datetimes_forecast_start)

    # get indexes for each column (ready for when convert to numpy)
    col_idxs = {col: i for i, col in enumerate(X_test.columns)}

    # get target values at the datetimes to be predicted
    y_test_at_predictions = y_test[y_test.index.isin(indexes_of_all_predictions)]

    if np.isnan(y_test_at_predictions).any():
        raise Exception('y true values have nans in them')
    # convert the test data to numpy
    X_test_array = X_test.to_numpy()
    y_test_array = y_test.to_numpy()

    ##############
    # main evaluation loop
    ##############

    # loop through each saved model, as well as two naive models, which just use features already computed
    for path_name in glob(str(MODELS_PATH/ '*')):
        
        # if model is LSTM then skip as evaluation dealt with seperately
        if 'LSTM' in path_name:
            continue
        # otherwise if not a file then skip
        if os.path.isfile(path_name) is False:
            continue
        else:       
            with open(path_name, 'rb') as f:
                # get model name
                model_name = Path(path_name).stem
                # if model is _not_ naive, then load the model from the path
                model = pkl.load(f)
        
        #############
        # get predictions
        ############
        
        # and fill in values
        y_preds_array = get_forecast(model_object=model, X_test_array=X_test_array, y_test_array=y_test_array, 
                                     initial_features_indexes=initial_features_indexes, forecast_horizon=HORIZON, 
                                     col_idxs=col_idxs, context_window=CONTEXT_WINDOW)
        
        # create frame with predictions and labels for given client id
        df_preds = pd.DataFrame(index=indexes_of_all_predictions, data={'y_pred': y_preds_array.flatten(),
                                                        'y_true': y_test_at_predictions})
        
        # now unscale the predictions and labels to get original units
        df_preds_unscaled = df_preds.groupby(level=
                                        'client_id').transform(lambda g: 
                                                                unscale_per_client(g, df_mean_std_usages_for_data))
    
        # for each client compute the rmse
        rmse_per_client = df_preds_unscaled.groupby(level=
                                                    'client_id').apply(lambda g: 
                                                                            root_mean_squared_error(g['y_true'], g['y_pred']))
        
        # test to make sure that rmse does not contain nans
        if np.isnan(rmse_per_client).any():
            raise Exception('rmse has nans')
        
        # normalise rmse by mean usage to make comparable across clients
        nrmse_per_client = rmse_per_client/df_mean_std_usages_for_data['mean_usage']


        # create dict of summary stats
        summary_dict = {
                        'mean': nrmse_per_client.mean(),
                        'std': nrmse_per_client.std(),
                        'max': nrmse_per_client.max(),
                        'min': nrmse_per_client.min(),
                        'top_5_performing_clients': list(nrmse_per_client.sort_values().iloc[:5].index),
                        'bottom_5_performing_clients': list(nrmse_per_client.sort_values(ascending=False).iloc[:5].index),
                        }

        # log summary stats
        log_stats(model_name=model_name, nrmse_summary_dict=summary_dict, nrmse_per_client=nrmse_per_client, df_preds=df_preds_unscaled)

        print(f'\n {model_name} predictions and performance metrics logged')


if __name__ == '__main__':
    evaluate_models()
        
