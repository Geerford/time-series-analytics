from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, STATUS_FAIL
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score


class Creator(ABC):
    @staticmethod
    @abstractmethod
    def train(params: dict, features: pd.DataFrame, target: pd.DataFrame) -> dict:
        pass

    @staticmethod
    @abstractmethod
    def optimize(features: pd.DataFrame, target: pd.DataFrame, max_evals: int, verbose=False):
        pass

    @staticmethod
    @abstractmethod
    def run(features, target, test_size: float = 0.2, max_evals: int = 15):
        pass

    @staticmethod
    def train_model(model, features: pd.DataFrame, target: pd.Series, **fit_parameters) -> dict:
        """
        Train a model with time series cross-validation by returning
        the right dictionary to be used by hyperopt for optimization

        :param model: a model implementing the standard scikit-learn interface
        :param features: pd.DataFrame holding the training features
        :param target: pd.Series holding the training target
        :param fit_parameters: dict with parameters to pass to the model fit function
        :return: Dictionary holding the resulting model, MAE and final status of the training
        as required by hyperopt interface
        """

        try:
            model.fit(features, target, **fit_parameters)

            scorer = make_scorer(mean_absolute_error, greater_is_better=False)
            cv_space = TimeSeriesSplit(n_splits=5)
            cv_score = cross_val_score(model, features, target.values.ravel(), cv=cv_space, scoring=scorer)

            return {
                "loss": np.mean(np.abs(cv_score)),
                "status": STATUS_OK,
                "model": model
            }

        except ValueError as ex:
            return {
                "error": ex,
                "status": STATUS_FAIL
            }

    @staticmethod
    def test_model(model, features: pd.DataFrame, target: pd.Series) -> dict:
        """
        Get the MAE for a given model on a test dataset

        :param model: a model implementing the standard scikit-learn interface
        :param features: pd.DataFrame holding the features of the test set
        :param target: pd.Series holding the test set target
        :return test_score: the MAE on the test dataset
        """

        predictions = model.predict(features)
        return mean_absolute_error(target.values, predictions)
