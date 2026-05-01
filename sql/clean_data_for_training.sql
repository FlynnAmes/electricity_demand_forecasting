-- Clean all the raw data, to create table of clean data that in this project, 
-- will just form the 'actual' values that will be piped into teh actuals table at each 
-- 24 hour interval (different to cleaning required in inference!)


CREATE OR REPLACE TABLE cleaned_training_hourly_usages AS (
-- CTE to melt the table, so that have a row for each timestamp and client id
WITH melt_table AS (

    UNPIVOT raw_data
    ON  COLUMNS(* EXCLUDE column000)
    INTO 
        NAME client_id
        VALUE usage
),

-- Then get hourly usage by averaging over three previous intervals
-- Use CASE statement here to give nulls where not enough preceding rows for full hour

-- Note that assumes complete 15 minute intervals (i.e, no missing intervals) for now
get_hourly_usage AS (

    SELECT  column000 AS recorded_at,
            client_id,
            CASE WHEN COUNT(*) OVER w = 4
-- average usage over the hour will give the hourly usage in Kwh
                 THEN AVG(usage) OVER w
                 ELSE NULL
            END AS hourly_usage

    FROM melt_table
    WINDOW w AS (
                PARTITION BY client_id
                ORDER BY recorded_at
                ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
                )
),

-- get usage at hour only, to get hourly usage data only
-- Note this effectively filters out Null values created by the earlier window operation
-- Then also convert client id string to a integer with prefix removed
get_usage_at_the_hour_and_convert_client_ids AS (

    SELECT  recorded_at,
            REPLACE(client_id, 'MT_', ''):: INTEGER AS client_id,
            hourly_usage
    FROM get_hourly_usage
    WHERE DATE_PART('minute', recorded_at) = 0

),

-- remove data where not a customer, first get the cumulative usage
get_cumu_usage AS (

    SELECT *,
    SUM(hourly_usage) OVER(PARTITION BY client_id
             ORDER BY recorded_at
             ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as cumu_usage
    FROM get_usage_at_the_hour_and_convert_client_ids
),

-- and remove where the cumulative usage is zero (assuming that not a customer at that point)
remove_where_non_customer AS (

    SELECT recorded_at,
           client_id,
           hourly_usage

    FROM get_cumu_usage
    WHERE cumu_usage > 1E-9
)

-- now finally just select all rows, so can create table using these
SELECT * FROM remove_where_non_customer
ORDER BY client_id, recorded_at

);


-- and explort to parquet file
COPY cleaned_training_hourly_usages TO '../data/processed/hourly_usage_cleaned.parquet';


-- now create another set of data containing only clients present in both training and testing 
-- periods (leeping simple here with simple time split)

-- create table for invalid clients
CREATE OR REPLACE TABLE invalid_clients (
    client_id INTEGER
);

-- clients with bad data identified via visual inspection (would do this with rules based approach ideally)
INSERT INTO invalid_clients (client_id)
VALUES
    ('196'),
    ('362'),
    ('279'),
    ('093'),
    ('223'),
    ('003'),
    ('223'),
    ('003'),
    ('332'),
    ('347'),
    ('001'),
    ('131'),
    ('015');

-- get training and testing split dates
WITH training_cutoff_date AS (
    SELECT MIN(recorded_at) + TO_HOURS(CEIL(DATE_DIFF('HOURS', MIN(recorded_at), MAX(recorded_at))*0.7)::INTEGER) AS training_cutoff
    FROM cleaned_training_hourly_usages
), 

-- note use of floor here, for conservative check
testing_cutoff_date AS (
    SELECT MIN(recorded_at) + TO_HOURS(FLOOR(DATE_DIFF('HOURS', MIN(recorded_at), MAX(recorded_at))*0.8)::INTEGER) AS training_cutoff
    FROM cleaned_training_hourly_usages
), 

-- get min and max dates that have data for each client
get_timeframe_of_data AS (
    SELECT client_id,
           MIN(recorded_at) as min_date,
           MAX(recorded_at) as max_date
    FROM cleaned_training_hourly_usages
    GROUP BY client_id
),

-- get invalid clients, defined as those who are not present in both the training and testing splits
get_invalid_clients AS (
    SELECT client_id
    FROM get_timeframe_of_data
-- clients must have data in both the training and testing subsets (so that can test them!)
    WHERE min_date > (SELECT * FROM training_cutoff_date) OR max_date < (SELECT * FROM testing_cutoff_date)
)

-- insert invalid clients into table
INSERT INTO invalid_clients
SELECT * FROM get_invalid_clients;

-- create table with additional filter of valid clients (needed for training/testing)
CREATE OR REPLACE TABLE cleaned_training_hourly_usages_valid_clients AS (
    -- save data with valid clients as well as the clients deemed to be invalid
    WITH remove_invalid_clients AS (

        SELECT *
        FROM cleaned_training_hourly_usages
        WHERE client_id NOT IN (SELECT * FROM invalid_clients)

    )

    SELECT * FROM remove_invalid_clients
);

-- save to parquet files
COPY hourly_usage_cleaned_valid_clients TO '../data/processed/hourly_usage_cleaned_valid_clients.parquet';
COPY invalid_clients TO '../data/processed/invalid_clients.json';
