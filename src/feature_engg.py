from imports import *


class feat_engg(object):
    """_summary_: This function is used to create new features from existing features.

    Parameters:
        df {dataframe}: A dataframe with the raw data
        config {dict}: A dictionary with the configuration parameters
    methods:
        categorify_columns(): A function to label encode the categorical columns

        count_encode_columns(): A function to take count (value_count) of the categorical columns.

        split_datetime_col(): A function to split the datetime columns into year, month, day for the column 'pickup_date'.

        cal_time_diff(): A function to calculate the time difference between pickup_date and query_date in days
    """

    def __init__(self, df, config):
        self.df = df
        self.config = config

    def categorify_columns(self):
        """_summary_: A function to label encode the categorical columns using categorify function from utils.
        parameters:
            None
        returns:
            df {dataframe}: A dataframe with the label encoded columns

        """
        for col in self.config["categorify_cols"]:
            self.df = categorify(df=self.df, cat=col, freq_treshhold=20)
        return self.df

    def count_encode_columns(self):
        """_summary_: A function to take count (value_count) of the categorical columns using count_encode function from utils.
        parameters:
            None
        returns:
            df {dataframe}: A dataframe with the count encoded columns
        """
        for col in self.config["count_encoded_cols"]:
            self.df = count_encode(self.df, col)
        return self.df

    def split_datetime_col(self):
        """_summary_: A function to split the datetime columns into year, month, day for the column 'pickup_date'.
        parameters:
            None
        returns:
            df {dataframe}: A dataframe with the split datetime columns
        """
        # Split datetime columns into year, month, day.
        for colname in self.df[self.config['date_cols']]:
            self.df[colname] = self.df[colname].astype('datetime64[ns]')
        # Using split_datetime function from utils.
        self.df = split_datetime(self.df, "pickup_date")
        return self.df

    def cal_time_diff(self):
        """_summary_: A function to calculate the time difference in days between pickup date and query date  columns.
        parameters:
            None
        returns:
            df {dataframe}: A dataframe with the time_to_pickup column.

        """
        # Calculate time difference between two datetime columns.
        self.df['time_to_pickup'] = self.df['pickup_date'] - \
            self.df['query_date']
        self.df['time_to_pickup'] = self.df['time_to_pickup'].dt.days
        return self.df
