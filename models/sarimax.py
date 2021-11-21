from functools import partial

import pandas as pd
from hyperopt import STATUS_OK, hp, Trials, tpe, fmin
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX

from models.creator import Creator


class SarimaxModel(Creator):
    @staticmethod
    def train(params: dict, features: pd.DataFrame, target: pd.DataFrame) -> dict:
        """
        Train a ARIMA model with given input parameters

        :param params: dict with parameters to be passed to model constructor
        :param features: pd.DataFrame holding the training features
        :param target: pd.DataFrame holding the training target
        :return: Dictionary holding the resulting model, MAE score and final status of the training
        as required by hyperopt interface
        """
        D = 0
        params = {
            'order': (int(params["p"]), int(params["d"]), int(params["q"])),
            'seasonal_order': (int(params["P"]), D, int(params["Q"]), int(params["s"])),
            'trend': params['trend'],
            'random_state': 21
        }
        model = SARIMAX(**params)

        return Creator.train_model(model, features, target, method='lbfgs')

    @staticmethod
    def optimize(features: pd.DataFrame, target: pd.DataFrame, max_evals: int, verbose=False) -> (dict, Trials):
        """
        Run Bayesan optimization to find the optimal XGBoost algorithm
        hyperparameters.

        :param features: pd.DataFrame with the training set features
        :param target: pd.DataFrame with the training set targets
        :param max_evals: the maximum number of iterations in the Bayesian optimization method
        :param verbose: if True print the best output parameters
        :return best: dict with the best parameters obtained
        :return trials: a list of hyperopt Trials objects with the history of the optimization
        """

        space = {
            'p': hp.uniform('p', 1, 20),
            'd': hp.uniform('d', 1, 20),
            'q': hp.uniform('q', 1, 20),
            'P': hp.uniform('P', 1, 5),
            'Q': hp.uniform('q', 1, 5),
            's': hp.uniform('s', 1, 20),
            'trend': hp.choice('trend', ['n', 'c', 't', 'ct']),
        }

        objective_fn = partial(SarimaxModel.train, features=features, target=target)

        trials = Trials()
        best = fmin(fn=objective_fn, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

        if verbose:
            print(f"""
            Best parameters:
                p: {best["p"]}  
                d: {best["d"]}
                q: {best['q']}
                P: {best["P"]}  
                Q: {best["Q"]}
                s: {best['s']}
                trend: {best['trend']}
            """)

        return best, trials

    @staticmethod
    def run(features, target, test_size: float = 0.2, max_evals: int = 15):
        """
        Full training and testing pipeline for XGBoost ML model with
        hyperparameter optimization using Bayesian method

        :param features: pd.DataFrame holding the model features
        :param target: pd.DataFrame holding the target values
        :param max_evals: maximum number of iterations in the optimization procedure
        :param test_size: number of test size
        :return model: the optimized CatBoost model
        :return cv_score: the average MAE coming from cross-validation
        :return test_score: the MAE on the test set
        """

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, shuffle=False)

        best, trials = SarimaxModel.optimize(X_train, y_train, max_evals=max_evals, verbose=True)
        model = SarimaxModel.train({}, X_train, y_train)["model"]
        cv_score = min([f["loss"] for f in trials.results if f["status"] == STATUS_OK])
        test_score = Creator.test_model(model, X_test, y_test)
        return model, cv_score, test_score
