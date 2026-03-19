## Overwiew

System to predict electricity demand an individual client, with a forecast horizon of 24 hours, and a context window of 1 week.

Uses electricity usage data from over 300 clients, obtainable <a href=https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014>here</a>.

Note: This project is in progress and will be committed once a draft of a deep learning model is 
up and running.


## Approach so far

- Rules based approaches to clean and preprocess input data

- Naive baseline evaluated using demand 1 day and week prior, as well as the average usage at that time, day-of-week & month across the data

- Multiple linear regression and elastic-net regression models created and evaluated to identify feature 
(lagged, rolling statistics as well as temporal features) importance. Performance evaluated using normalised root mean squared error.

- XGBoost model implemented to compare to above and deep learning approaches.


## Next steps

- Currently implementing LSTM model in PyTorch. Will compare to statistical and ML baselines. Will then implement and compare a transformer-based architecture.

- Experiment with anomaly detection techniques (e.g., isolation forest) to identify anomalous input data, as alternative to rules based approaches.

- For now, a global normalisation is applied in preprocessing. Switch to cluster based normalisation and check if improves performance (e.g., allowing DL models to better focus upon variations within client data 
rather than between them). 

- Evaluate performance within clusters rather than globally (to identify which clients get best/worst performance, to inform future improvements)

- Decide upon the model to use in production. Implement automated testing, containerisation, API and logging of inferences.