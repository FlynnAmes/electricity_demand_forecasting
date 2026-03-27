""" initial cleaning, distilling (converting from 15 Kw to hourly Kwh) and melting of data, to 
get in format ready for time series forecasting """

import pandas as pd
import numpy as np
from paths import DATA_PATH


def clean_distill_melt_data():

    # load in raw data. account for European decimals being commas, and data being in ; seperated txt file
    df = pd.read_csv(str(DATA_PATH / 'raw' / 'LD2011_2014.txt'), sep=';', decimal=',')

    print('shape before changes:', df.shape)


    ############
    # dropping bad columns
    ############

    # drop 362 which looks like aggregate. Also 196 who's usage looks suspicously high currently.
    # also 279 who seems to have had an instrumentation error later on. and 3 for similar reasons
    # also 93 for suspcous looking peaks. and 223 who seems to stop being customer or data fails after a while
    # also has anomalous data point.
    # 347 has constant usage at all times, then dissapearing for year before coming back etc.
    # 332 has a instantaneuos 10x magnitude jump shift midway through data that looks like instrument change/issue
    # 001 also has two changepoints
    # 015 has huge absence of data part way through the sequence
    #  #TODO  handle this columns more gracefully (e.g., imputation, 
    # keeping of ealier data)
    # #TODO use proper arules for automatic cleansing (as want to make scalable)!
    # e.g., certain instances of zero energy usage, anomaly detection techniques, (e.g., isolation forest),
    # CUSUM for changepoints
    df.drop(columns=['MT_196', 'MT_362', 'MT_279', 'MT_093', 'MT_223', 'MT_003', 'MT_332', 'MT_347', 
                     'MT_001', 'MT_131', 'MT_015'], inplace=True)

    # rename first column to datetime
    df.rename(columns={'Unnamed: 0': 'datetime'}, inplace=True)

    # now convert the new column to datetime format
    df['datetime'] = pd.to_datetime(df['datetime'])

    # # and rop final MT_370 columns which looks to be a composite sum of the other columns
    # df.drop(columns=['MT_370'], inplace=True)

    print('shape after changes:', df.shape)

    # for now, condense dataset by getting energy consumed in the previous hour
    # first need to convert to Kwh:
    df_kwh_adjust = df.copy()

    # index using df.columns to change all columns except the datetime. divide by four and sum 
    # with previous three rows to get the hourly energy usage in Kwh 
    df_kwh_adjust.loc[:, df.columns != 'datetime'] = (df_kwh_adjust.loc[:, df.columns != 'datetime']/4).rolling(window=4).sum()

    # now remove nan columns from start of dataset (as none present elsewhere)
    df_kwh_adjust.dropna(axis=0, inplace=True)

    # and only maintain hourly data for the time being (to condense for now)
    # need to use datetime accessor to get the date attributes
    df_hourly_data = df_kwh_adjust[df_kwh_adjust['datetime'].dt.minute == 0]

    # now to reorientate data so that have datetime and the client id (initially columns) as row definers
    df_melted = df_hourly_data.melt(id_vars=['datetime'])

    # and rename columns so more meaningful
    df_melted.rename(columns={'variable': 'client_id',
                            'value': 'hourly_usage_kwh'}, inplace=True)

    # convert client id to integer (i.e., remove letters) (TODO: this operation inefficient but fine for now)
    df_melted['client_id'] = df_melted['client_id'].apply(lambda str: np.int64(str.replace('MT_', '')))

    # get cumulative usage
    df_melted['cumsum_usage'] = df_melted.groupby('client_id')['hourly_usage_kwh'].transform('cumsum')

    print('shape before clipping', df_melted.shape)
    # then use cumulayive usage to remove data. For now keeping simple, 
    # removing all rows where the cumulative usage is zero, assuming that all data before is
    # because they were not yet a customer (possible that could have been on holiday at data start)
    # but data also started in January so considered less likely here. (note employing tolerance)
    df_melted_clipped = df_melted[df_melted['cumsum_usage'] > 1e-9]

    print('shape after clipping', df_melted_clipped.shape)

    # now drop the cumsum column which not using for now
    df_melted_clipped.drop(columns=['cumsum_usage'], inplace=True)

    # and save to csv file (processed)
    #TODO: do small change so that index column not made into a column by default
    # index = false gets rid of unnamed column
    df_melted_clipped.to_csv(str(DATA_PATH / 'processed' / 'hourly_usage_cleaned.csv'), index=False)


if __name__ == '__main__':
    clean_distill_melt_data()