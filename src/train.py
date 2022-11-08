from imports import *


class train_model(object):

    def __init__(self, df, final_cols, config):
        self.config = config
        self.df = df
        self.final_cols = final_cols

    def split_data(self):
        """
        __summary__: Split the data into train and test set. Perform SMOTE on training data to handle class imbalance.

        params: self.df{pd.DataFrame}: Dataframe with the data.

        returns: X_train{pd.Series}: 80% of the data for training.
                 y_train{pd.Series}: 80% of the data for training.
                 X_test{pd.Series}: 20% of the data for testing.
                 y_test{pd.Series}: 20% of the data for testing.
        """
        # Split the data into train and test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df[self.final_cols], self.df.target, test_size=0.2, random_state=42)
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(
            self.X_train, self.y_train)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def base_model(self):
        """
        __summary__: Train the base model with the default parameters.

        params: self.X_train{pd.Series}: 80% of the data for training.
                self.y_train{pd.Series}: 80% of the data for training.

        returns: rf_model{object}: Trained random forest model.
        """
        # Train the base model with default parameters
        logistic_reg = LogisticRegression(
            solver='sag', max_iter=1000, random_state=42)  # use sag solver for large datasets
        logistic_reg.fit(self.X_train, self.y_train)

        return logistic_reg

    def hyperparameter_tuning_randomforest(self) -> None:
        """
        __summary__: Function for Hyperparameter tuning for random forest model.

        returns: best_params{dict}: Dictionary of best parameters for random forest model.

        """
        # hyperparameter tuning for random forest.
        self.rf_model = RandomForestClassifier()
        scoring = {'f1': 'f1', 'precision': 'precision', 'recall': 'recall'}
        print('Performing Randomized Search CV for Random Forest')
        self.random_search = RandomizedSearchCV(
            self.rf_model, self.config["parameter_grid_rf"]["param_grid"], scoring=scoring,
            refit="f1", cv=2, verbose=2, n_jobs=-1, return_train_score=True)
        self.random_search.fit(self.df[self.final_cols],
                               self.df.target)
        self.best_params = self.random_search.best_params_

    def train_random_forest(self) -> object:
        """
        __summary__: Train the random forest model with the best parameters from hyperparameter tuning.

        params: self.best_params{dict}: Dictionary of best parameters for random forest model.
                self.X_train{pd.Series}: 80% of the data for training.
                self.y_train{pd.Series}: 80% of the data for training.

        returns: rf_model{object}: Trained random forest model.
        """
        # Get the best parameters from the grid search and train the model
        randomforest = RandomForestClassifier(
            **self.best_params, oob_score=True)
        randomforest.set_params(**self.best_params)
        randomforest.fit(self.X_train, self.y_train)
        return randomforest
