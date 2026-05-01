""" Train linear regression (OLS and Lasso) and tree-based (XGBoost) models """

import pickle as pkl
from sklearn.linear_model import LinearRegression, Lasso
import yaml
from paths import DATA_PATH, MODELS_PATH, CONFIG_PATH
from xgboost import XGBRegressor
import os
import pandas as pd
import numpy as np


def save_model(model_object, model_name: str):

    """ save model object """

    # check it exists
    os.makedirs(MODELS_PATH, exist_ok=True)

    # then save 
    with open(MODELS_PATH / (model_name + '.pkl'), 'wb') as f:
        pkl.dump(model_object, f)


def get_train_validation_data():

    """ Load in the feature data and return the train, validation splits for features and target,
     given specified training and validation sizes (as portion of total data).
    Validation is performed on data later in time vs training etc."""

    # load from parquet
    df_train = pd.read_parquet(DATA_PATH / 'processed' / 'df_tabular_train.parquet').set_index(['client_id', 'target_time'])
    df_validate = pd.read_parquet(DATA_PATH / 'processed' / 'df_tabular_validation.parquet').set_index(['client_id', 'target_time'])

    # split into features ans target
    X_train = df_train.drop(columns=['target_hourly_usage'])
    X_validate = df_validate.drop(columns=['target_hourly_usage'])

    y_train = df_train['target_hourly_usage']
    y_validate = df_validate['target_hourly_usage']
   
    return X_train, X_validate, y_train, y_validate


def train_non_seq_models():
    #####################
    # get config params
    #####################

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # hyperparameters for models
    Lasso_params = config['Lasso_params']
    XGBoost_params = config['XGBoost_params']

    ############
    # load data
    ############

    X_train, X_validate, y_train, y_validate = get_train_validation_data()

    print('train & validation split complete')

    # create dictionary of model objects (with hyperparameters specified)
    model_dict = {'OLS': LinearRegression(),
                'Lasso': Lasso(**Lasso_params),
                'XGBoost': XGBRegressor(**XGBoost_params)}

    # loop through models to fit them
    for model_name, model_object in model_dict.items():

        ############
        # Fit the model
        ############

        # if XGBoost, implementing early stopping using validation data
        if model_name == 'XGBoost':
            model_object.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_validate, y_validate)])
        
        # otherwise a regular fit
        else:
            model_object.fit(X_train, y_train)
        
        print(f'\n {model_name} fitted')

        ############
        # save the model
        ############
        save_model(model_object=model_object, model_name=model_name)

        print(f'\n {model_name} model object saved')


if __name__ == '__main__':
    train_non_seq_models()
