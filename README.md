## Overwiew

System to predict multi-client electricity demand, with a forecast horizon of 24 hours, and a context window of 1 week.

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

The electricity usage data (for over 300 clients) is obtainable <a href=https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014>here</a>.


## Approach so far

- Naive baseline evaluated using lagged usage from prior hour and week. 

- Linear models (OLS and Lasso), tree-based (XGBoost) and deep learning sequential (LSTM) models trained  evaluated

- Per client-normalisation upon input data, to ensure model pays equal attention to residential and industrial clients (whose magnitdue can be an order of magnitude different).

- Cyclical encoding of time features. Linear and tree-based models also use lagged and rolling statistics

- Evaluation performed using client-mean-normalised rmse


## Results (preliminary) ##

Client-mean normalised root mean square error (NRMSE) summary stats:

```
────────────────────────────────────────────────────
MODEL           | mean NRMSE | max NRMSE | min NRMSE
────────────────────────────────────────────────────
Naive 1wk lag   | 0.15       | 0.66      | 0.04
── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── 
LSTM            | 0.12       | 0.72      | 0.02
── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── 
Lasso           | 0.11       | 0.65      | 0.02
── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── 
OLS             | 0.10       | 0.67      | 0.02
── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── 
XGBoost         | 0.08       | 0.46      | 0.02
────────────────────────────────────────────────────
```

<br>

Tree-based and linear models currently outpeform the deep learning model. This is because the problem
is predominantly autoregressive. Linear models accounting for lag features can handle an autoregressive problem well (e.g., note in the Lasso implementation here, the only with coeffs not driven to zero are 1hr, 1dy and 1wk lag usages. Note similar performance to XGBoost).

A small subset of clients drive a large portion of the nrmse


## Next steps

- Rule based approaches/anomaly detection for data cleansing, to improve scalability

- Hyperparameter tuning with Optuna

- Multi-timestep prediction