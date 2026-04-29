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
COPY cleaned_training_hourly_usages TO '../data/processed/hourly_usage_cleaned.parquet'
