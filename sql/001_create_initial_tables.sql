-- Load raw data into a table. This way, schema are inferred automatically (with enforce later?)
-- note that path assumes that running file from the db directory!
CREATE TABLE raw_data AS 
    SELECT * FROM read_csv('../data/raw/LD2011_2014.txt', 
                            delim=';',
                            decimal_separator=',',
                            sample_size=-1);

------------------------------------------------------

-- create table for cleaned data
CREATE TABLE cleaned_data(

client_id INTEGER,
recorded_at DATETIME,
hourly_usage DOUBLE,
PRIMARY KEY (client_id, recorded_at)

);

-------------------------------------------------------

-- table for raw inference data. Data will be appended to the table in daily intervals 
CREATE TABLE raw_data_inference AS
    FROM raw_data
    LIMIT 0;

------------------------------------------------------

-- table for cleaned inference data. Data will be appended to the table in daily intervals
CREATE TABLE cleaned_data_inference AS
    FROM cleaned_data
    LIMIT 0;

-------------------------------------------------------

-- create table for features (generated from clean data)
CREATE TABLE feature_store (

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
CREATE TABLE predicted_usage (

client_id INTEGER,
recorded_at DATETIME,
hourly_usage DOUBLE,

PRIMARY KEY (client_id, recorded_at)

);

--------------------------------------------------

-- contains the actual values of usage (update as data is obtained each day)
CREATE TABLE actual_usage (

client_id INTEGER,
recorded_at DATETIME,
hourly_usage DOUBLE,

PRIMARY KEY (client_id, recorded_at)

);