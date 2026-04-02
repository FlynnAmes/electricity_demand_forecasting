""" Tune XGBoost model using Optuna. Save these hyperparameters to use in final training """

import optuna
import xgboost as xgb
import pickle as pkl
from paths import DATA_PATH, LOGS_PATH, FIGURES_PATH, CONFIG_PATH
from sklearn.metrics import root_mean_squared_error
import json
import yaml


def get_dmatrices():
     
    # get the training and validation data (already standardised)
    with open(DATA_PATH / 'processed' / 'df_tabular_train.pkl', 'rb') as f:
        df_train = pkl.load(f)

    with open(DATA_PATH / 'processed' / 'df_tabular_validate.pkl', 'rb') as f:
        df_validate = pkl.load(f)

    # seperate into features and target
    X_train = df_train.drop(columns=['hourly_usage_kwh'])
    X_validate = df_validate.drop(columns=['hourly_usage_kwh'])

    y_train = df_train['hourly_usage_kwh']
    y_validate = df_validate['hourly_usage_kwh']

    # convert training and validation to native XGBoost DMatrix format
    d_train = xgb.DMatrix(X_train, label=y_train)
    d_validate = xgb.DMatrix(X_validate, label=y_validate)

    return d_train, d_validate, y_validate


def objective(trial):

        # first suggest parameters (early stopping rounds is kept constant)
        # and define those which not tuning (e.g., eval metric)
        params_dict = {
        'learning_rate' : trial.suggest_float('lr', 5e-3, 5e-1, log=True),
        'max_depth' : trial.suggest_int('max_depth', 3, 10),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 10),
        'subsample' : trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_lambda' : trial.suggest_float('lambda', 0.0, 1.0),
        'eval_metric': 'rmse',
                    }

        # callback used to prune unpromising trials (checked at end of each boosting round)
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'validation-rmse') 

        # train model (note need for num_boost_round rather than n_estimators)
        model = xgb.train(params_dict, dtrain=d_train, evals=[(d_validate, 'validation')], callbacks=[pruning_callback], 
                          num_boost_round=200, early_stopping_rounds=10)

        # get predictions form fitted model to compute rmse
        y_pred = model.predict(d_validate)

        # return rmse for configuration
        return root_mean_squared_error(y_true=y_validate, y_pred=y_pred)


def log_params(study):
    """ save best hyperparameters and parameter importance"""

    # save parameters from best trial ('best hyperparameters)
    with open(LOGS_PATH / 'XGBoost' / 'best_params.json', 'w') as f:
        json.dump(study.best_trial.params, f, indent=4)

    # save hyperparameter importances
    with open(LOGS_PATH / 'XGBoost' / 'param_importances.json', 'w') as f:
        json.dump(optuna.importance.get_param_importances(study), f, indent=4)


def save_plots(study):
    """ save plots from the tuning process """

    # hyperparameter importance
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(FIGURES_PATH / 'xgb_hyperparam_importance.png')

    # and optimisation history
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(FIGURES_PATH / 'xgb_optuna_opt_history.png')



if __name__ == '__main__':
    
    try:
        # load the config
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        
        # get params for XGBoost fitting
        num_boost_round = config['XGBoost_params']['n_estimators']
        early_stopping_rounds = config['XGBoost_params']['early_stop_rounds']
        optuna_params = config['optuna_params']

        # get d matrices, and validation data for loss computation
        d_train, d_validate, y_validate = get_dmatrices()

        # set up the study and initiate it
        study = optuna.create_study(direction='minimize') 
        study.optimize(objective, **optuna_params)

    except KeyboardInterrupt:
        print('\n tuning interupted by user')

    except Exception as e:
        print(f'tuning failed with error {e}')
        raise

    # always log and plot
    finally:
        # log best params, parameter importance etc. and plot them
        log_params(study)
        print('\n hyperparameters logged')
        save_plots(study)
        print('\n plots saved')