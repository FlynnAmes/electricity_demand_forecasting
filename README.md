## Overwiew

System to predict electricity demand an individual client, with a forecast horizon of 24 hours, and a context window of 1 week.

Uses electricity usage data from over 300 clients, obtainable <a href=https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014>here</a>.

Note: This project is in progress and not yet finished!!


## Approach so far


- Naive baseline evaluated using demand 1 day and week prior, as well as the average usage at that time, day-of-week & month across the data

- Multiple linear regression and elastic-net regression models created and evaluated to identify feature 
(lagged, rolling statistics as well as temporal features) importance. Performance evaluated using normalised root mean squared error.

- XGBoost model implemented to compare to above and deep learning approaches.

- Initial draft LSTM implemented in PyTorch (see notebooks/LSTM.ipynb)


## Current limitations

- cleansing of bad data performed using visual inspection - not scalable. Will replace with rules based approaches and anomaly detection techniques (e.g., isolation forest)

- Normalisation performed globally rather than per-client (meaning model may focus 
too much on differences in magnitude between clients, rather than within profile).
Will attempt cluster based normalisation.

- Use of embeddings in LSTM requires exisiting client data. No metadata available in training data to 
create embeddings (instead learned during LSTM training). For inference with with new clients, may need to use cluster-based average for during a 'warm-up' period where initial data is collected.


## Next steps

- Finish implementation of LSTM

- Makes client data that using consistent between XGBoost and LSTM. Implement early stopping for XGBoost

- Tune hyperparameters of models via Optuna

- Cluster based normalisation.

- Evaluate performance within clusters rather than globally (to identify which clients get best/worst performance, to inform future improvements)

- Evaluate performance and produce graphs of time series. 

- Implement transformer based forecasting model and compare to existing models.

- Decide upon the model to use in production. Implement automated testing, containerisation, API and logging of inferences.