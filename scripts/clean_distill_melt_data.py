""" initial cleaning, distilling (converting from 15 Kw to hourly Kwh) and melting of data, to 
get in format ready for time series forecasting """

import pandas as pd
import numpy as np
from paths import DATA_PATH


def clean_distill_melt_data():

    # load in raw data. account for European decimals being commas, and data being in ; seperated txt file
    df = pd.read_csv(str(DATA_PATH / 'raw' / 'LD2011_2014.txt'), sep=';', decimal=',')

    print('shape before removing bad clients:', df.shape)


    ############
    # dropping bad columns
    ############

    # drop 362 which looks like aggregate. Also 196 who's usage looks suspicously high.
    # also 279 who looks to have had an instrumentation error later on. 3 for similar reasons.
    # 93 for suspcous looking peaks. 223 who seems to either stop being a customer, or data fails after a while
    # 347 has constant usage at all times, then dissapearing for year before coming back etc.
    # 332 has a instantaneuos 10x magnitude jump shift midway through data that looks like instrument change/issue
    # 001 also has two changepoints
    # 015 has huge absence of data part way through the sequence

    # NOTE that will get round to using proper arules for automatic cleansing (to make scalable)!
    # e.g., certain instances of zero energy usage,  # CUSUM for changepoints,
    # anomaly detection techniques, (e.g., isolation forest),
    df.drop(columns=['MT_196', 'MT_362', 'MT_279', 'MT_093', 'MT_223', 'MT_003', 'MT_332', 'MT_347', 
                     'MT_001', 'MT_131', 'MT_015'], inplace=True)

    # rename first column to datetime
    df.rename(columns={'Unnamed: 0': 'datetime'}, inplace=True)

    # convert to datetime format
    df['datetime'] = pd.to_datetime(df['datetime'])

    print('\n shape after removing bad clients:', df.shape)

    ###############
    # for now, condense dataset by getting energy consumed in the previous hour
    ###############

    # first need to convert to Kwh:
    df_kwh_adjust = df.copy()

    # divide usage by 4 and sum  with previous three rows to get the hourly energy usage in Kwh (avoiding the datetime column)
    df_kwh_adjust.loc[:, df.columns != 'datetime'] = (df_kwh_adjust.loc[:, df.columns != 'datetime']/4).rolling(window=4).sum()

    # remove nan columns generated at start of dataset where no precedning rows
    df_kwh_adjust.dropna(axis=0, inplace=True)

    # only maintain data on the hour (to condense for now)
    df_hourly_data = df_kwh_adjust[df_kwh_adjust['datetime'].dt.minute == 0]

    # reorientate data so that have datetime and the client id (initially columns) as row unique identifier
    df_melted = df_hourly_data.melt(id_vars=['datetime'])

    # rename columns so more meaningful
    df_melted.rename(columns={'variable': 'client_id',
                            'value': 'hourly_usage_kwh'}, inplace=True)

    # convert client id to integer (i.e., remove letters)
    df_melted['client_id'] = df_melted['client_id'].apply(lambda str: np.int64(str.replace('MT_', '')))

    # get cumulative usage (below)
    df_melted['cumsum_usage'] = df_melted.groupby('client_id')['hourly_usage_kwh'].transform('cumsum')

    print('shape before clipping', df_melted.shape)

    # Use cumulative usage to remove data where client not yet a customer.
    # removing all rows where the cumulative usage is zero, assuming that all data before usage accumulates
    # is where not yet a customer (employing tolerance)
    df_melted_clipped = df_melted[df_melted['cumsum_usage'] > 1e-9]

    print('shape after clipping', df_melted_clipped.shape)

    # now drop the cumsum column which no longer using
    df_melted_clipped.drop(columns=['cumsum_usage'], inplace=True)

    # and save cleaned data to csv
    df_melted_clipped.to_csv(DATA_PATH / 'processed' / 'hourly_usage_cleaned.csv', index=False)


if __name__ == '__main__':
    clean_distill_melt_data()