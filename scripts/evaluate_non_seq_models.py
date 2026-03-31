""" evaluate performance of naive, linear, and tree based models upon testing data """

import pandas as pd
import pickle as pkl
from sklearn.metrics import root_mean_squared_error
import numpy as np
import json
from paths import DATA_PATH, MODELS_PATH, LOGS_PATH
from glob import glob
from pathlib import Path
import os


def unscale_per_client(group, dict_with_mean_and_std):
    """ use mean and std usage for each client to unscale data, ready for 
    evaluation """

    client_id = int(np.unique(group.index.get_level_values('client_id'))[0])

    return ((group * dict_with_mean_and_std[client_id]['std']) + dict_with_mean_and_std[client_id]['mean'])


def log_stats(model_name: str, nrmse_per_client, nrmse_summary_dict, df_preds):
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
    with open(DATA_PATH / 'processed' / f'{model_name}_preds.pkl', 'wb') as f:
        pkl.dump(df_preds, f)


def evaluate_models():

    ############
    # load test data
    ############

    with open(DATA_PATH / 'processed' / 'df_tabular_test.pkl', 'rb') as f:
        df_test = pkl.load(f)

    # split into features and target
    X_test = df_test.drop(columns=['hourly_usage_kwh'])
    y_test = df_test['hourly_usage_kwh']

    ############
    # load in mean and std usages for each client.
    ############

    # for unscaling labels and predictions
    with open(DATA_PATH / 'processed' / 'mean_std_per_client.json', 'r') as f:
        df_std_mean_usage = pd.read_json(f).T
        # create dictionary version for fast lookup
        dict_std_mean_usage = df_std_mean_usage.to_dict(orient='index')


    ##############
    # main evaluation loop
    ##############

    # loop through each saved model, as well as two naive models, which just use features already computed
    for path_name in glob(str(MODELS_PATH/ '*')) + ['naive_1hr', 'naive_1wk']:

        # if model is naive, do nothing, otherwise load in model from path  
        if 'naive' in path_name:
            model_name = path_name
        else:       
            with open(path_name, 'rb') as f:
                # get model name
                model_name = Path(path_name).stem
                # if model is _not_ naive, then load the model from the path
                model = pkl.load(f)

        # if model is naive, then just use appropriate feature
        if model_name == 'naive_1hr':
            # just use usage from hour prior
            y_pred = X_test['lag_1hr']
        elif model_name == 'naive_1wk':
            # using usage from week prior
            y_pred = X_test['lag_1hr']
        else:
            # otherwise if not naive, fit the model using test data
            y_pred = model.predict(X_test)


        print(f'\n {model_name} predictions made')


        # create frame with predictions and labels for given client id
        df_preds = pd.DataFrame(index=df_test.index, data={'y_pred': y_pred,
                                                        'y_true': y_test})
        
        # now unscale the predictions and labels to get original units
        df_preds_unscaled = df_preds.groupby(level=
                                        'client_id').transform(lambda g: 
                                                                unscale_per_client(g, dict_std_mean_usage))
        
        # for each client compute the rmse
        rmse_per_client = df_preds_unscaled.groupby(level=
                                                    'client_id').apply(lambda g: 
                                                                            root_mean_squared_error(g['y_true'], g['y_pred']))
        
        # normalise rmse by mean usage to make comparable across clients
        nrmse_per_client = rmse_per_client/df_std_mean_usage['mean']

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
        log_stats(model_name=model_name, nrmse_summary_dict=summary_dict, nrmse_per_client=nrmse_per_client, df_preds=df_preds)

        print(f'\n {model_name} predictions and performance metrics logged')


if __name__ == '__main__':
    evaluate_models()