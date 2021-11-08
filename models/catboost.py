from functools import partial

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from hyperopt import STATUS_OK, hp, Trials, tpe, fmin
from sklearn.model_selection import train_test_split

from models.creator import Creator


class CatboostModel(Creator):
    @staticmethod
    def train(params: dict, features: pd.DataFrame, target: pd.DataFrame) -> dict:
        """
        Train a Catboost model with given input parameters

        :param params: dict with parameters to be passed to model constructor
        :param features: pd.DataFrame holding the training features
        :param target: pd.DataFrame holding the training target
        :return: Dictionary holding the resulting model, MAE score and final status of the training
        as required by hyperopt interface
        """

        params = {
            'loss_function': 'RMSE',
            'eval_metric': 'MAE',
            'early_stopping_rounds': 300,
            'random_seed': 21,
            'boosting_type': 'Ordered',  # 'Plain'
            'verbose': False,
            'task_type': 'CPU',

            'n_estimators': int(params["n_estimators"]),
            'learning_rate': params["learning_rate"],
            'subsample': params['subsample'],
            # 'max_depth': int(params["max_depth"])
        }
        model = CatBoostRegressor(**params)

        return Creator.train_model(model, features, target,
                                   eval_set=[(features, target.values.ravel())],
                                   early_stopping_rounds=params['early_stopping_rounds'],
                                   verbose=params['verbose'])

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
            'n_estimators': hp.quniform('n_estimators', 200, 1200, 100),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.1)),
            'subsample': hp.uniform('subsample', 0.8, 1)
            # "max_depth": hp.quniform("max_depth", 3, 14, 1),
            # 'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
            # 'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
            # 'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
            # 'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
            # 'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
        }

        objective_fn = partial(CatboostModel.train, features=features, target=target)

        trials = Trials()
        best = fmin(fn=objective_fn, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

        if verbose:
            print(f"""
            Best parameters:
                learning_rate: {best["learning_rate"]}  
                n_estimators: {best["n_estimators"]}
                subsample: {best['subsample']}
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

        # best = {
        #     'learning_rate': 0.02,
        #     'n_estimators': 600,
        #     'subsample': 0.95
        # }
        # cv_score = None
        best, trials = CatboostModel.optimize(X_train, y_train, max_evals=max_evals, verbose=True)
        model = CatboostModel.train(best, X_train, y_train)["model"]
        cv_score = min([f["loss"] for f in trials.results if f["status"] == STATUS_OK])
        test_score = Creator.test_model(model, X_test, y_test)
        return model, cv_score, test_score
