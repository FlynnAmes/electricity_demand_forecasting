## Overwiew

System to predict multi-client electricity demand, with a forecast horizon of 24 hours, and a context window of 1 week.

Uses electricity usage data from over 300 clients, obtainable <a href=https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014>here</a>.

Note: This project is in progress and not yet finished.


## Approach so far

- Naive baseline evaluated using demand 1 day and week prior, as well as the average usage at that time, day-of-week across the data.

- OLS and Lasso regression models, along with XGBoost, implemented and evaluated to benchmark deep learning approach with linear and tree based models. Using Lagged and rolled features, normalised on a per-client basis (rather than globally, to ensure models focus upon variation within client profiles, rather than clients with 
the largest magnitude usage)

- Deep learning sequential model (LSTM) implemented and evaluated.


## Current limitations

- cleansing of bad data performed using visual inspection - not scalable. Will replace with rules based approaches and anomaly detection techniques (e.g., isolation forest)

- code is in notebooks and not yet productionised.

## Next steps

- Evaluate performance across client clusters (not just globally) to identify for which clients models perform well/not well

- Hyperparameter tuning with Optuna

- Rule based approaches/anomaly detection for data cleansing, to improve scalability

- Implement one more deep learning model (transformer-based) to compare vs LSTM and existing models.

- Decide upon the model to use in production. Implement automated testing, containerisation, API and logging of inferences.