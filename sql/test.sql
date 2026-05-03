SELECT DISTINCT DATE_PART('WEEKDAY', recorded_at) AS weekday 
FROM cleaned_training_hourly_usages_valid_clients 
LIMIT 10