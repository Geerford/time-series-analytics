import warnings
from functools import partial

import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, hp, Trials, tpe, fmin, STATUS_FAIL
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

from models.creator import Creator


class CustomSARIMAX(BaseEstimator):
    def __init__(self, order, seasonal_order, trend):
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.model = None

    def fit(self, target, method='lbfgs', disp=False, maxiter=25):
        self.model = SARIMAX(
            target,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend
        ).fit(method=method, disp=disp, maxiter=maxiter)

    def score(self, target):
        return -mean_absolute_error(target, self.model.forecast(len(target)))

    @staticmethod
    def train(model, target, **fit_parameters):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model.fit(target, **fit_parameters)

                cv_space = TimeSeriesSplit(n_splits=5)
                cv_score = cross_val_score(model, target.values.ravel(), cv=cv_space)
            return {
                "loss": np.mean(np.abs(cv_score)),
                "status": STATUS_OK,
                "model": model.model
            }
        except ValueError as ex:
            return {
                "error": ex,
                "status": STATUS_FAIL
            }

    @staticmethod
    def test(model, target):
        predictions = model.forecast(len(target))
        return mean_absolute_error(target.values, predictions)


class SarimaxModel(Creator):
    @staticmethod
    def train(params: dict, target: pd.DataFrame, **kwargs) -> dict:
        """
        Train a ARIMA model with given input parameters

        :param params: dict with parameters to be passed to model constructor
        :param target: pd.DataFrame holding the training target
        :return: Dictionary holding the resulting model, MAE score and final status of the training
        as required by hyperopt interface
        """
        params = {
            'order': (int(params["p"]), int(params["d"]), int(params["q"])),
            'seasonal_order': (int(params["P"]), int(params["D"]), int(params["Q"]), int(params["s"])),
            'trend': params['trend']
        }
        model = CustomSARIMAX(**params)

        return CustomSARIMAX.train(model, target, method='lbfgs', disp=False, maxiter=1)

    @staticmethod
    def optimize(target: pd.DataFrame, max_evals: int, verbose=False, **kwargs) -> (dict, Trials):
        """
        Run Bayesan optimization to find the optimal XGBoost algorithm
        hyperparameters.

        :param target: pd.DataFrame with the training set targets
        :param max_evals: the maximum number of iterations in the Bayesian optimization method
        :param verbose: if True print the best output parameters
        :return best: dict with the best parameters obtained
        :return trials: a list of hyperopt Trials objects with the history of the optimization
        """

        space = {
            'p': hp.uniform('p', 1, 3),
            'd': hp.uniform('d', 0, 2),
            'q': hp.uniform('q', 1, 3),
            'P': hp.uniform('P', 1, 6),
            'D': hp.uniform('D', 0, 2),
            'Q': hp.uniform('Q', 1, 6),
            's': hp.uniform('s', 1, 12),
            'trend': hp.choice('trend', ['n', 'c', 't', 'ct'])
        }

        objective_fn = partial(SarimaxModel.train, target=target)

        trials = Trials()
        best = fmin(fn=objective_fn, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials, return_argmin=False)

        if verbose:
            print(f"""
            Best parameters:
                p: {int(best["p"])}  
                d: {int(best["d"])}
                q: {int(best['q'])}
                P: {int(best["P"])}
                D: {int(best["D"])}  
                Q: {int(best["Q"])}
                s: {int(best['s'])}
                trend: {best['trend']}
            """)

        return best, trials

    @staticmethod
    def run(target, test_size: float = 0.2, max_evals: int = 15, **kwargs):
        """
        Full training and testing pipeline for XGBoost ML model with
        hyperparameter optimization using Bayesian method

        :param target: pd.DataFrame holding the target values
        :param max_evals: maximum number of iterations in the optimization procedure
        :param test_size: number of test size
        :return model: the optimized CatBoost model
        :return cv_score: the average MAE coming from cross-validation
        :return test_score: the MAE on the test set
        """

        y_train, y_test = train_test_split(target, test_size=test_size, shuffle=False)

        best, trials = SarimaxModel.optimize(y_train, max_evals=max_evals, verbose=True)
        model = SarimaxModel.train(best, target)["model"]
        cv_score = min([f["loss"] for f in trials.results if f["status"] == STATUS_OK])
        test_score = CustomSARIMAX.test(model, y_test)
        return model, cv_score, test_score
