""" create and train pipelines for sklearn & XGBoost models"""

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
import pickle as pkl
from scipy import stats
import numpy as np
from paths import DATA_PATH, MODELS_PATH, LOGS_PATH, CONFIG_PATH
import yaml
import os
import json
from datetime import datetime


def save_model(rgr, model_name: str):
    """ saves the ML model to the models directory in pickle format, using model name (along 
     with MODELS PATH defined in paths.py) """

    # create directory if doesn't already exists
    os.makedirs(MODELS_PATH, exist_ok=True)

    # and save the model to this directory
    with open(MODELS_PATH / (model_name + '.pkl'), 'wb') as f:
        pkl.dump(rgr, f)


def log_training_params(rgr, model_name: str):
    """ Logs hyperparameters of sklearn estimators, as well as
      cross validation results obtained during training, and feature names for the XGboost
      Logs are saved using the current date, along with a name given to the model, to 
      the logs/ directory

      rgr: the trained and tuned model pipeline RandomSearchCV object
      model_name: the name of the model (string), used for saving """
    
    # get hyperparameters
    pipeline_params = rgr.best_estimator_.get_params()

    # then convert to string where not simple python object, so can serialise into json
    # (nested loop fine here as not many params)
    pipeline_params_reformat = {k: str(var) if not isinstance(var, (int, float, str, bool, type(None))) 
                                else var for k, var in pipeline_params.items()}

    # For now different versions distinusguished using datetime
    savedir = LOGS_PATH / 'model_development' / model_name / 'pretuning' / datetime.now().strftime('%Y%m%d')

    # if dir name does not exist, make it
    os.makedirs(savedir, exist_ok=True)

    # dump to JSON
    with open(savedir / 'hyperparams.json', 'w') as f:
        json.dump(pipeline_params_reformat, f, indent=4)
    
    # also dump the random search cv results - to csv for now
    pd.DataFrame(data=rgr.cv_results_).to_csv(savedir / 'cv_results.csv')


def train_models():

    # load in config params
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    random_seed = config['random_seed']
    n_splits_time = config['n_splits_time']
    scoring_training = config['scoring_training']
    n_iter_r_search = config['n_iter_r_search']
    en_max_iter = config['en_max_iter']


    # load the data
    with open(DATA_PATH / 'processed' / 'X_train.pkl', 'rb') as f:
        X_train = pkl.load(f)

    with open(DATA_PATH / 'processed' /  'y_train.pkl', 'rb') as f:
        y_train = pkl.load(f)

    ###########
    # create pipeline dictionary for models, param grids for hyperparameter tuning
    ###########

    pipeline_dict = {
                    'OLS': {'pipeline': Pipeline([('scale', StandardScaler()), ('clf', LinearRegression())]),
                            },

                    'OLS_PCA': {'pipeline': Pipeline([('scale', StandardScaler()), ('PCA', PCA()), ('clf', LinearRegression())]),
                                },

                    
                    'xgb': {'pipeline': Pipeline([('scale', StandardScaler()), ('clf', XGBRegressor())]),
                                },

                    'elastic_net': {'pipeline': Pipeline([('scale', StandardScaler()), ('clf', ElasticNet(max_iter=en_max_iter))]),
                                    },

                    'elastic_net_PCA': {'pipeline': Pipeline([('scale', StandardScaler()), ('PCA', PCA()), ('clf', ElasticNet(max_iter=en_max_iter))]),
                                        },
    }

    # dummy grid for OLS as no params to tune (wrap value in list to make iterable)
    pipeline_dict['OLS']['param_grid'] = {'clf__fit_intercept': [True]}
    pipeline_dict['OLS_PCA']['param_grid'] = {'clf__fit_intercept': [True],
                                              # using float ensure retains enough components to explain x percentage of features
                                              'PCA__n_components': [0.7, 0.8, 0.9, 0.95]
                                             }

    # for elastic net
    pipeline_dict['elastic_net']['param_grid']  = {'clf__alpha': stats.loguniform(1e-2, 1e2),
                                                'clf__l1_ratio': stats.uniform()}
    pipeline_dict['elastic_net_PCA']['param_grid']  = {'clf__alpha': stats.loguniform(1e-2, 1e2),
                                                'clf__l1_ratio': stats.uniform(),
                                                'PCA__n_components': [0.7, 0.8, 0.9, 0.95]}


    # for XGBoost
    pipeline_dict['xgb']['param_grid'] = {'clf__learning_rate': stats.loguniform(0.01, 1), 
                                'clf__n_estimators': np.linspace(100, 2000, 10, dtype=np.int64),
                                'clf__max_depth': np.linspace(2, 20, 10, dtype=np.int64), 
                                'clf__gamma': np.linspace(0, 5, 6, dtype=np.float64),
                                'clf__subsample': np.linspace(0.1, 1, 10, dtype=np.float64),
                                'clf__colsample_bytree': np.linspace(0.1, 1, 10, dtype=np.float64),
                                'clf__reg_alpha': np.linspace(0, 1, 11, dtype=np.float64),
                                'clf__reg_lambda': np.linspace(0, 1, 11, dtype=np.float64),   
        }

    ##############
    # fitting the models
    ##############

    # set up time series splitter - this splits model to ensure that testing data can only come after training 
    # in time. NOTE that uses rows to split. By ordering using datetime as the primary key, ensures that split 
    # by datetime and not by client
    tscv = TimeSeriesSplit(n_splits=n_splits_time)

    for model_name, config in pipeline_dict.items():

        print(f'\n model to fit: {model_name}')

        # set up the random search object
        rgr = RandomizedSearchCV(estimator=config['pipeline'],
                                param_distributions=config['param_grid'], cv=tscv, 
                                scoring=scoring_training, random_state=random_seed, n_iter=n_iter_r_search)
        
        # fit the random search CV to the data
        rgr.fit(X=X_train, y=y_train)

        print(f'\n model fitted')

        #TODO: can use the model objects to evaluate feature importance later

        # save the model
        save_model(rgr, model_name=model_name)

        print(f'\n model saved')

        # log params
        log_training_params(rgr, model_name=model_name)


if __name__ == '__main__':
    train_models()





