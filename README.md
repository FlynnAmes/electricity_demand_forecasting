## Overwiew

System to predict multi-client electricity demand, with a 24 hour prediction horizon and 1 week historical context window.

The project employs classical statistical, machine learning, and deep learning approaches to understand which methods perform best for multi-client time series data.

## Approach

- Implemented a naive baseline using seasonal lagged usage (1 hour, 1 day, 1 week)

- Trained and evaluated:
    - Linear models (OLS, Lasso)
    - Tree-based models (XGBoost)
    - Sequential deep learning model (LSTM, PyTorch)

- Applied per-client normalisation to handle scale differences across clients

- Engineered time-based features using cyclical encoding

- Used lagged values and rolling statistics for non-sequential models
 
- Performed time-based train/validation/test splits
 
- Evaluated performance using client-mean-normalised RMSE (NRMSE)


## Key Findings ##

```
MODEL           | mean NRMSE
-----------------------------
Naive (1wk lag) | 0.15
LSTM            | 0.12
Lasso           | 0.11
OLS             | 0.10
XGBoost         | 0.08
```

- Tree-based and linear models outperform the LSTM, despite the higher capacity of the latter

- Lasso selects only three features (1-hour, 1-day, 1-week lags), indicating the problem is low-dimensional and strongly autoregressive

- XGBoost captures nonlinear interactions, providing a modest improvement over linear models

- Performance varies across clients. Those with highest coefficient of variation exhibit the largest errors,
suggesting increased variance in usage increases forecasting difficulty


These results suggest:

- Most of the predictive signal is contained within a small selection of lag features

- The problem is largely linearly autoregressive, limiting the benefit of deep learning sequence models



## Per-client NRMSE distributions

![Per-client NRMSE distribution](plots/client_error_histogram.png)


## Repository Structure

```
├── data/
│   ├── processed/
│   └── raw/   
│  
├── logs/                          # training and evaluation logs (gitignored)
├── models/                        # trained model artifacts (gitignored)
├── notebooks/                     # exploratory analysis/model creation
│
├── scripts/                       # training and evaluation code
│   ├── classes.py
│   ├── paths.py
│   ├── clean_data.py
│   ├── prep_features.py
│   ├── prep_sequential.py
│   ├── train_non_seq_models.py
│   ├── train_LSTM.py
│   ├── evaluate_non_seq_models.py
│   └── evaluate_LSTM.py
│   
├── config.yml
├── .gitignore
└──  README.md
```

## Data ##

The dataset (~300 clients) is obtainable here: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014


## Extensions

- Anomaly detection for data cleansing and validation.

- Hyperparameter tuning with Optuna.

- Multi-timestep forecasting (sequence-to-sequence models).