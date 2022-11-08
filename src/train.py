from imports import *


class train_model(object):

    def __init__(self, X_train, y_train, X_test, y_test, config):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.config = config

    def base_model(self):
        """
        __summary__: Train the base model with the default parameters.

        params: self.X_train{pd.Series}: 80% of the data for training.
                self.y_train{pd.Series}: 80% of the data for training.

        returns: rf_model{object}: Trained random forest model.
        """
        # Train the base model with default parameters
        logistic_reg = LogisticRegression(
            solver='sag', max_iter=1000, random_state=42)
        logistic_reg.fit(self.X_train, self.y_train)
        # save the model to disk

        pickle.dump(logistic_reg, open(os.path.join(
            self.config['PATHS']['Project_path'] + 'models/', self.config['base_model_name_lr']), "wb"), compress=3)
        return logistic_reg

    def hyperparameter_tuning_randomforest(self) -> None:
        """
        __summary__: Function for Hyperparameter tuning for random forest model.

        returns: best_params{dict}: Dictionary of best parameters for random forest model.

        """
        self.logger.info("Hyperparameter tuning for random forest model")
        # hyperparameter tuning for random forest.
        self.rf_model = RandomForestClassifier()
        self.random_search = RandomizedSearchCV(
            self.rf_model, self.config["parameter_grid_rf"]["param_grid"], scoring=['f1', 'roc_auc'], cv=5, verbose=2, n_jobs=-1)
        self.random_search.fit(self.df[self.final_columns],
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
        joblib.dump(randomforest, open(os.path.join(
            self.config['PATHS']['Project_path'] + 'models/', self.config['model_name_random_forest']), "wb"), compress=3)  # compress the model to reduce the size of the model
        return randomforest
