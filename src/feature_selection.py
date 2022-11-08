from imports import *


class feature_selection(object):
    """_summary_: A class to select the features for the model
    Parameters:
        df {dataframe}: A dataframe with the raw data
        config {dict}: A dictionary with the configuration parameters

    methods:
        cont_feature_mutual_info(): A function to select the continuous features using one-way ANOVA test.
        cont_feature_chi2(): A function to select the continuous features using chi2
        cat_feature(): A function to select the categorical features using mutual_info_classif
    """

    def __init__(self, df, config):
        self.df = df
        self.config = config

    def cont_feature_oneway_anova(self):
        """
        _summary_: A function to select the continuous features using one-way ANOVA test. 
                    The method uses selectKBest from sklearn.feature_selection to select the features. 
                    The k value is set to 7 which is the number of features to be selected.
                    This method is based on F-test estimate the degree of linear dependency between two random variables.
                    Returns k highest scoring features based on the F-test.

        Returns:
            select_features {list}: Returns list of select features from the index of the selected features from selectKBest function
        """
        fs = SelectKBest(score_func=f_classif, k=7)
        X = self.df[self.config['cont_cols_for_feature_selection']]
        fs.fit_transform(X, self.df['target'])
        select_features = X.columns[fs.get_support()]
        return select_features

    def cont_feature_chi2(self):
        """_summary_ : A function to select the categorical features using chi-squared stats.
                    The method uses selectKBest from sklearn.feature_selection to select the features.
                     n_features features with the highest values for the test chi-squared statistic from X,
                    which must contain only non-negative features such as booleans or frequencies relative to the classes.

        Returns:
            select_features {list}: Returns list of select features from the index of the selected features from selectKBest function
        """
        self.df[self.config['cat_cols_for_feature_selection']
                ] = self.df[self.config['cat_cols_for_feature_selection']].astype('int64')
        chi_features = SelectKBest(chi2, k=7)
        X = self.df[self.config['cat_cols_for_feature_selection']]
        y = self.df['target']
        x_best = chi_features.fit_transform(X, y)
        select_features = X.columns[chi_features.get_support()]
        return select_features

    def cat_feature_mutual_info(self):
        """_summary_: A function to select the categorical features using mutual_info_classif.
            The best features are selected based on best mutual information score for the categorical target variable.
            The method uses selectKBest from sklearn.feature_selection to select the features.
            The k value is set to 7 which is the number of features to be selected.
        Returns:
            select_features {list}: Returns list of select features from the index of the selected features from selectKBest function
        """
        self.df[self.config['cat_cols_for_feature_selection']
                ] = self.df[self.config['cat_cols_for_feature_selection']].astype('int64')
        fs = SelectKBest(score_func=mutual_info_classif, k=7)
        X = self.df[self.config['cat_cols_for_feature_selection']]
        y = self.df['target']
        fs.fit_transform(X, y)
        select_features = X.columns[fs.get_support()]
        return select_features
