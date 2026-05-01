-- creates all features used for training data

-- create a table with clients to omit


CREATE OR REPLACE TABLE mean_std_usages AS (

-- for now, hard code the train split for the data (will use dbt to input configuration parameters later)
WITH training_cutoff_date AS (
    SELECT MIN(recorded_at) + TO_HOURS(CEIL(DATE_DIFF('HOURS', MIN(recorded_at), MAX(recorded_at))*0.7)::INTEGER) AS training_cutoff
    FROM cleaned_training_hourly_usages_valid_clients
),

-- get mean and standard deviation of usages - use for normalisation later
get_mean_std_usages AS (

    SELECT client_id,
           AVG(hourly_usage) as mean_usage,
           STDDEV(hourly_usage) as std_usage
    FROM   cleaned_training_hourly_usages_valid_clients
-- making sure that only apply upon the training data!
    WHERE recorded_at < (SELECT * FROM training_cutoff_date)
    GROUP BY client_id 
)

SELECT *
FROM get_mean_std_usages 

);


CREATE OR REPLACE TABLE features_for_training AS (

    WITH normalise_usages AS (
    -- now normalise whole of each client data using per-client mean and standard deviation
        SELECT clh.client_id,
            clh.recorded_at,
            (clh.hourly_usage - mean_usage)/std_usage as hourly_usage_normalised
        FROM 
        cleaned_training_hourly_usages_valid_clients AS clh
        INNER JOIN mean_std_usages AS msu
        ON clh.client_id = msu.client_id
    ),

    compute_features AS (

        -- wrap in subquery so can filter out nulls
        SELECT * FROM (

            SELECT clh.client_id,
                clh.recorded_at as target_time,
                clh.hourly_usage_normalised as target_hourly_usage,

        -- encode the time features
        -- hour
                SIN(2*PI()*DATE_PART('HOUR', clh.recorded_at)/24) as hour_sin,
                COS(2*PI()*DATE_PART('HOUR', clh.recorded_at)/24) as hour_cos,
        -- day of week
                SIN(2*PI()*DATE_PART('WEEKDAY', clh.recorded_at)/7) as day_sin,
                COS(2*PI()*DATE_PART('WEEKDAY', clh.recorded_at)/7) as day_cos,
        -- month of year
                SIN(2*PI()*DATE_PART('MONTH', clh.recorded_at)/12) as month_sin,
                COS(2*PI()*DATE_PART('MONTH', clh.recorded_at)/12) as month_cos,


        -- get lag features
                LAG(hourly_usage_normalised, 1) OVER(w_lag) AS lag_1hr,
                LAG(hourly_usage_normalised, 2) OVER(w_lag) AS lag_2hr,
                LAG(hourly_usage_normalised, 6) OVER(w_lag) AS lag_6hr,
                LAG(hourly_usage_normalised, 24) OVER(w_lag) AS lag_1dy,
                LAG(hourly_usage_normalised, 168) OVER(w_lag) AS lag_1wk,

        -- get rolling features - make sure that gives null if not enough values to get rolling time window
                CASE WHEN COUNT(*) OVER day_roll = 24 THEN
                    MIN(hourly_usage_normalised) OVER day_roll
                ELSE NULL END AS rolling_min_1dy,

                CASE WHEN COUNT(*) OVER day_roll = 24 THEN
                    MAX(hourly_usage_normalised) OVER day_roll
                ELSE NULL END AS rolling_max_1dy,

                CASE WHEN COUNT(*) OVER day_roll = 24 THEN
                    AVG(hourly_usage_normalised) OVER day_roll
                ELSE NULL END AS rolling_mean_1dy,

                CASE WHEN COUNT(*) OVER day_roll = 24 THEN
                    STDDEV(hourly_usage_normalised) OVER day_roll 
                ELSE NULL END AS rolling_std_1dy,

                CASE WHEN COUNT(*) OVER week_roll = 168 THEN
                    MIN(hourly_usage_normalised) OVER week_roll
                ELSE NULL END AS rolling_min_1wk,

                CASE WHEN COUNT(*) OVER week_roll = 168 THEN
                    MAX(hourly_usage_normalised) OVER week_roll
                ELSE NULL END AS rolling_max_1wk,

                CASE WHEN COUNT(*) OVER week_roll = 168 THEN
                    AVG(hourly_usage_normalised) OVER week_roll
                ELSE NULL END AS rolling_mean_1wk,

                CASE WHEN COUNT(*) OVER week_roll = 168 THEN
                    STDDEV(hourly_usage_normalised) OVER week_roll
                ELSE NULL END AS rolling_std_1wk,

                mean_usage

            FROM normalise_usages AS clh
            INNER JOIN mean_std_usages AS msu
            ON clh.client_id = msu.client_id

            WINDOW w_lag AS (
                PARTITION BY clh.client_id
                ORDER BY recorded_at
            ),

            day_roll AS (
                PARTITION BY clh.client_id
                ORDER BY recorded_at
                ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING
            ),

            week_roll AS (
                PARTITION BY clh.client_id
                ORDER BY recorded_at
                ROWS BETWEEN 168 PRECEDING AND 1 PRECEDING
            )
        )
    -- filter rows using the largest lag column - if null here then will capture nulls 
    -- created in all other columns where not enough preceding values
        WHERE lag_1wk IS NOT NULL
    )

    SELECT *
    FROM compute_features 
    ORDER BY client_id, target_time

    );


-- save the full feature dataset to a parquet file
COPY features_for_training TO '../data/processed/features_for_training.parquet';
-- and also save the mean and std usages for each client
COPY mean_std_usages TO '../data/processed/mean_std_usages_per_client.json';


-- create random subset of clients (to manage computational cost)
CREATE OR REPLACE TABLE clients_in_training AS (

    WITH random_client_subset AS (

        SELECT client_id
    -- select from client ids only! (because feature data has many clients)
    -- distinct doesn't guarantee same order so order clients in subquery, to ensure 
    -- sample is same each time when using a fixed seed
        FROM (SELECT DISTINCT client_id FROM features_for_training
                    ORDER BY client_id)
    -- hard code number of clients for now (will want to pass in from config!)
        USING SAMPLE reservoir(100 ROWS)
    -- assign random seed (again to be passed through as config parameter)
        REPEATABLE (42)
        ORDER BY client_id
    )

    SELECT * 
    FROM random_client_subset
);

-- save the clients used in training to a json
COPY clients_in_training TO '../data/processed/client_subset_for_training.json';

------------
-- split data into training, validation and testing
------------

-- create variables for the training and validation cutoff dates (so can then reference in later queries)

-- get training split date
SET VARIABLE  training_cutoff_date = (
    SELECT MIN(recorded_at) + TO_HOURS(CEIL(DATE_DIFF('HOURS', MIN(recorded_at), MAX(recorded_at))*0.7)::INTEGER) AS training_cutoff
    FROM cleaned_training_hourly_usages
);

-- get validation split date
SET VARIABLE validation_cutoff_date = (
    SELECT MIN(recorded_at) + TO_HOURS(CEIL(DATE_DIFF('HOURS', MIN(recorded_at), MAX(recorded_at))*0.8)::INTEGER) AS validation_cutoff
    FROM cleaned_training_hourly_usages
);

-- split and save the training, validation and testing data

-- training data
COPY (
   SELECT *
   FROM features_for_training
   WHERE target_time < getvariable('training_cutoff_date')
   AND client_id IN (SELECT * FROM clients_in_training) ) TO '../data/processed/df_tabular_train.parquet';

-- validation data
COPY (SELECT *
    FROM features_for_training
    WHERE target_time > getvariable('training_cutoff_date')
    AND target_time < getvariable('validation_cutoff_date')
    AND client_id IN (SELECT * FROM clients_in_training) ) TO '../data/processed/df_tabular_validation.parquet';

-- testing data
COPY (SELECT *
    FROM features_for_training
    WHERE target_time > getvariable('validation_cutoff_date')
    AND client_id IN (SELECT * FROM clients_in_training) ) TO '../data/processed/df_tabular_test.parquet';
