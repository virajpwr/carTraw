import yaml
import pandas as pd
import numpy as np


def load_config(file_path: str) -> dict:
    """
    __summary__: This function is used to load the config file.
    parameters:
        file_path {str} -- [path to the config file]
    returns:
        config {dict} -- [config file]
    """
    with open(file_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def categorify(df: pd.DataFrame, cat: str, freq_treshhold=20, unkown_id=1, lowfrequency_id=0) -> pd.DataFrame:
    """__summary__: This function is used perform encoding categorical features. 
    A frequency threshold is used to replace the categories with low frequency with a single category. 
    To deal with high cardinality and overfitting, we will replace the categories with low frequency with a single category.
    he category Ids 0 or 1 for a placeholder for the low frequency and unkown category.
    parameters: 
        df {pd.DataFrame} -- [dataframe]
        cat {str} -- [name of the categorical column]
        freq_treshhold {int} -- [frequency threshold]
        unkown_id {int} -- [to fil nan values]
        lowfrequency_id {int} -- [to replace low frequency categories]

    """
    freq = df[cat].value_counts()  # frequency of each category
    freq = freq.reset_index()  # reset the index

    freq.columns = [cat, 'count']  # rename the columns
    freq = freq.reset_index()  # reset the index

    freq.columns = [cat + '_Categorify', cat, 'count']  # rename the columns
    freq[cat + '_Categorify'] = freq[cat +
                                     '_Categorify']+2  # add 2 to the index
    freq.loc[freq['count'] < freq_treshhold,
             cat + '_Categorify'] = lowfrequency_id  # replace low frequency categories with 0

    freq = freq.drop('count', axis=1)  # drop the count column
    # merge the frequency dataframe with the original dataframe
    df = df.merge(freq, how='left', on=cat)
    # fill nan values with 1
    df[cat + '_Categorify'] = df[cat + '_Categorify'].fillna(unkown_id)
    # convert the column to category type
    df[cat + '_Categorify'] = df[cat + '_Categorify'].astype('category')
    return df


def count_encode(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """"
    __summary__: This function is used to perform count encoding on the categorical features. Count encoding done due to high cardinality of categorical columns.
                It calculates the frequency from one or more categorical features.
    Count Encoding (CE) calculates the frequency from one or more categorical features given the training dataset.

    Count Encoding creates a new feature, which can be used by the model for training.
     It groups categorical values based on the frequency together.
     Count Encoding creates a new feature, which can be used by the model for training. It groups categorical values based on the frequency together.
    parameters:
        df {pd.DataFrame} -- [dataframe]
        col {str} -- [name of the categorical column]
    returns:
        df {pd.DataFrame} -- [dataframe with new column]
    """
    # We keep the original order as cudf merge will not preserve the original order
    df['org_sorting'] = np.arange(len(df), dtype="int32")

    # count the number of each category
    df_tmp = df[col].value_counts().reset_index()  # reset the index
    df_tmp.columns = [col,  'CE_' + col]  # rename the columns with CE prefix
    df_tmp = df[[col, 'org_sorting']].merge(
        df_tmp, how='left', left_on=col, right_on=col).sort_values('org_sorting')  # merge the count with the original dataframe
    # fill the missing values with 0

    df['CE_' + col] = df_tmp['CE_' + col].fillna(0).values
    df = df.drop('org_sorting', axis=1)  # drop the temporary column
    # convert the column to int32
    # convert the column to int32
    df['CE_' + col] = df['CE_' + col].astype('int32')
    return df

def split_datetime(df: pd.DataFrame, colname: str) -> pd.DataFrame:
    """_summary_: This function is used to split the datetime column into separate columns.

    parameters:
        df {pd.DataFrame} -- [dataframe]
        colname {str} -- [name of the datetime column]
    returns:
        df {pd.DataFrame} -- [dataframe with new columns]
    """
    df['ts_weekday'] = df[colname].dt.weekday
    df['ts_day'] = df[colname].dt.day
    df['ts_month'] = df[colname].dt.month
    return df
