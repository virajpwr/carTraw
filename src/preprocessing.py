from imports import *


class preprocessing(object):
    """_summary_: A class to preprocess the data
    parameters:
        df {dataframe}: A dataframe with the raw data
        config {dict}: A dictionary with the configuration

    Methods:
        rename_cols(): Rename the columns based on the document for the task. Get the required columns from the config file.

        fillna(): Fill the missing values in the column 'cont_car_feature_4' with the mean of the column.

        remove_outliers(): Remove the outliers from the price column for target = 0 and target = 1 using IQR.

        drop_duplicates(): Drop the duplicates from the price column.

        convert_dtypes(): Convert the duration column to int64 and the categorical columns to category.

        labelencode(): Label encode the car_feature_9 and query_feature_3 columns.
    """

    def __init__(self, df, config):
        self.df = df
        self.config = config

    def rename_cols(self):
        self.df.columns = self.config['required_cols']

    def fillna(self):
        self.df['cont_car_feature_4'].fillna(
            self.df['cont_car_feature_4'].mean(), inplace=True)
        return self.df

    def remove_outliers(self):
        # remove outliers from the price column for target = 0 using IQR
        q1 = self.df[self.df['target'] == 0]['price'].quantile(0.25)
        q3 = self.df[self.df['target'] == 0]['price'].quantile(0.75)
        iqr = q3 - q1
        self.df = self.df[~((self.df['target'] == 0) & (self.df['price'] < (
            q1 - 1.5 * iqr)) | (self.df['price'] > (q3 + 1.5 * iqr)))]

        # remove outliers from the price column for target = 1 using IQR
        q1_1 = self.df[self.df['target'] == 1]['price'].quantile(0.25)
        q3_1 = self.df[self.df['target'] == 1]['price'].quantile(0.75)
        iqr_1 = q3_1 - q1_1
        self.df = self.df[~((self.df['target'] == 1) & (self.df['price'] < (
            q1_1 - 1.5 * iqr_1)) | (self.df['price'] > (q3_1 + 1.5 * iqr_1)))]
        return self.df

    def drop_duplicates(self):
        self.df = self.df.sort_values(by='price', ascending=False)
        self.df = self.df.drop_duplicates(subset=['price'])
        return self.df

    def convert_dtypes(self):
        self.df['duration'] = self.df['duration'].astype('int64')
        self.df[self.config['cat_cols']
                ] = self.df[self.config['cat_cols']].astype('category')
        return self.df

    def labelencode(self):
        le = LabelEncoder()
        self.df['car_feature_9'] = le.fit_transform(self.df['car_feature_9'])
        self.df['query_feature_3'] = le.fit_transform(
            self.df['query_feature_3'])
        return self.df
