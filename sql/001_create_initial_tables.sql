-- Load raw data into a table. This way, schema are inferred automatically (with enforce later?)
-- note that path assumes that running file from the db directory!
CREATE OR REPLACE TABLE raw_data AS 
    SELECT * FROM read_csv('../data/raw/LD2011_2014.txt', 
                            delim=';',
                            decimal_separator=',',
                            sample_size=-1);

-------------------------------------------------------

-- table for raw inference data. Data will be appended to the table in daily intervals 
CREATE OR REPLACE TABLE raw_data_inference AS
    FROM raw_data
    LIMIT 0;

------------------------------------------------------

-- table for cleaned inference data. Data will be appended to the table in daily intervals
CREATE OR REPLACE  TABLE cleaned_hourly_usage_inference(

client_id INTEGER,
recorded_at DATETIME,
hourly_usage DOUBLE,
PRIMARY KEY (client_id, recorded_at) 

);

-------------------------------------------------------

-- create table for features (generated from clean data)
CREATE OR REPLACE TABLE feature_store (

client_id INTEGER,
recorded_at DATETIME,

-- lag features
lag_1hr DOUBLE,
lag_2hr DOUBLE,
lag_6hr DOUBLE,
lag_1dy DOUBLE,
lag_1wk DOUBLE,

-- rolling features
rolling_mean_1dy DOUBLE,
rolling_mean_1wk DOUBLE,

rolling_max_1dy DOUBLE,
rolling_max_1wk DOUBLE,

rolling_min_1dy DOUBLE,
rolling_min_1wk DOUBLE,

rolling_std_1dy DOUBLE,
rolling_std_1wk DOUBLE,

-- extra features
-- mean usage of client over all time (in training data)
mean_usage DOUBLE,

PRIMARY KEY (client_id, recorded_at)

);

-------------------------------------------------

-- contains predictions made by models
CREATE OR REPLACE TABLE predicted_hourly_usage (

client_id INTEGER,
recorded_at DATETIME,
hourly_usage DOUBLE,

PRIMARY KEY (client_id, recorded_at)

);

--------------------------------------------------

-- contains the actual values of usage (update as data is obtained each day)
CREATE OR REPLACE TABLE actual_hourly_usage (

client_id INTEGER,
recorded_at DATETIME,
hourly_usage DOUBLE,

PRIMARY KEY (client_id, recorded_at)

);