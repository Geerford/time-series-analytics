import datetime
import os

import numpy as np
import pandas as pd

from decorator import Decorator
from describer import Describer
from forecaster import Forecaster
from models.sarimax import SarimaxModel
from models.catboost import CatboostModel
from transformer import Transformer
from utils import plot_go


class Model:
    def __init__(self, df: pd.DataFrame, plan: pd.Series, weekends: pd.DataFrame, params: dict, model_type: str):
        """
        Constructor for the Model object

        :param df: pd.DataFrame holding the input time-series target
        :param plan: pd.Series holding the planned sum over year
        :param weekends: pd.DataFrame of weekends and work days
        :param params: dict of any params
        :param model_type: Name of model
        """

        self.seed = params['seed']
        self.params = params
        self.plan = plan
        self.weekends = weekends
        self.model_type = model_type

        self.decorator = Decorator(weekends=self.weekends)
        self.transformer = Transformer(scale=True, diff=False, log=False, detrend=False)
        self.target = self.transformer.transform(df[['first_difference']])

        Describer.describe(self.target, period=self.params['period'])

    @staticmethod
    def model_switcher(model_type: str = "catboost"):
        """
        Forecasting over months from the 10th day of current month to the 10th day of next month

        :param model_type: name of model class
        :return: metaclass for defining abstract base classes
        """

        return {
            'catboost': CatboostModel,
            'sarimax': SarimaxModel
        }.get(model_type, 'Invalid type')

    def forecast(self, start_date: datetime.date, end_date: datetime.date, num_prediction: int) -> pd.DataFrame:
        """
        Forecasting over months from the 10th day of current month to the 10th day of next month

        :param start_date: start date of forecasting time period
        :param end_date: end date of forecasting time period
        :param num_prediction: number of forecasting as forecasting horizon
        :return: pd.DataFrame with forecast
        """

        period = f'{start_date}_{end_date}'
        print(f"[INFO] ID: {self.params['id']} forecast {period}")

        # Choose {num_prediction} time periods as forecasting horizon
        features, target = self.decorator.generate_features(df=self.target.loc[:end_date],
                                                            n_lags=num_prediction)

        X_train_i, y_train_i = features.loc[:start_date], target.loc[:start_date]
        X_test_i = features.loc[start_date + pd.Timedelta(days=1):end_date]
        y_test_i = target.loc[start_date + pd.Timedelta(days=1):end_date]

        model_i, cv_score, test_score = Model.model_switcher(self.model_type).run(X_train_i, y_train_i, max_evals=10)
        # Choose {num_prediction} time periods (~days) as forecasting horizon
        forecaster = Forecaster(weekends=self.weekends, num_prediction=num_prediction, decorator=self.decorator)

        y_pred_i = pd.DataFrame(model_i.predict(X_test_i), index=X_test_i.index).rename(columns={0: 't'})
        y_fc_i = pd.DataFrame(forecaster.recursive(y_train_i, model_i)).rename(columns={0: 't'})

        work_days_ind = self.weekends.loc[y_test_i.index].loc[self.weekends.loc[y_test_i.index]['flag'] == 0]
        y_test_i = self.transformer.inverse(y_test_i)
        y_pred_i = self.transformer.inverse(y_pred_i)
        y_fc_i = self.transformer.inverse(y_fc_i)

        model_name = f'cvr_{self.params["id"]}_{period}_diff'
        csv_path = f"{self.params['path_metrics']}metrics_pred.csv"
        os.makedirs(self.params['path_metrics'], exist_ok=True)
        os.makedirs(self.params['path_plots'], exist_ok=True)
        os.makedirs(self.params["path_models"], exist_ok=True)

        # Prediction metrics based on X_test
        self.decorator.calculate_metrics(actual=y_test_i.loc[work_days_ind.index]['t'].values,
                                         prediction=y_pred_i.loc[work_days_ind.index]['t'].values,
                                         plot_name=model_name + "_pred", silent=True,
                                         plot_path=self.params['path_plots'], csv_path=csv_path)

        csv_path = f"{self.params['path_metrics']}metrics_fc.csv"
        result_path = f"{self.params['path_metrics']}{period}"

        # Forecast metrics based on differenced time series
        fc_metrics_diff = self.decorator.calculate_metrics(actual=y_test_i.loc[work_days_ind.index]['t'].values,
                                                           prediction=y_fc_i.loc[work_days_ind.index]['t'].values,
                                                           plot_name=model_name + "_fc_diff", silent=True,
                                                           plot_path=self.params['path_plots'], csv_path=csv_path,
                                                           result_path=result_path + '_diff.csv')
        y_test_i_cumsum = y_test_i.loc[work_days_ind.index].groupby(pd.Grouper(freq='Y')).cumsum()
        y_fc_i_cumsum = y_fc_i.loc[work_days_ind.index].groupby(pd.Grouper(freq='Y')).cumsum()

        # Forecast metrics based on cumsum time series
        fc_metrics_cumsum = self.decorator.calculate_metrics(actual=y_test_i_cumsum.loc[work_days_ind.index]['t'].values,
                                                             prediction=y_fc_i_cumsum.loc[work_days_ind.index]['t'].values,
                                                             plot_name=model_name + "_fc_cumsum", silent=True,
                                                             plot_path=self.params['path_plots'], csv_path=csv_path,
                                                             result_path=result_path + '_cumsum.csv')

        model_i.save_model(f'{self.params["path_models"]}{model_name}.cbm')

        plot_go(traces=[
            [y_pred_i.index, y_pred_i['t'], 'y_pred_diff'],
            [y_fc_i.index, y_fc_i['t'], 'y_fc_diff'],
            [y_test_i.index, y_test_i['t'], 'y_test_diff'],
            [y_pred_i.index, y_pred_i.groupby(pd.Grouper(freq='Y')).cumsum()['t'], 'y_pred_cumsum'],
            [y_fc_i.index, y_fc_i.groupby(pd.Grouper(freq='Y')).cumsum()['t'], 'y_fc_cumsum'],
            [y_test_i.index, y_test_i.groupby(pd.Grouper(freq='Y')).cumsum()['t'], 'y_test_cumsum'],
        ],
            title=f'cvr_{self.params["id"]}_{period} Forecast',
            save_path=f'{self.params["path_plots"]}{model_name}.png', silent=True)

        work_days_only = target[['t']].loc[
            self.weekends.loc[target[['t']].index].loc[
                self.weekends.loc[target[['t']].index]['flag'] == 0].index]

        result = pd.DataFrame({
            'cvr': self.params["id"],
            'docdate': y_fc_i.index,
            'forecast': y_fc_i['t'],
            'forecastdate': start_date,
            'MAE_diff': fc_metrics_diff['MAE'],
            'MAPE_diff': fc_metrics_diff['MAPE'],
            'MaxAE_diff': fc_metrics_diff['MaxAE'],
            'SMAPE_diff': fc_metrics_diff['SMAPE'],
            'MAE_cumsum': fc_metrics_cumsum['MAE'],
            'MAPE_cumsum': fc_metrics_cumsum['MAPE'],
            'MaxAE_cumsum': fc_metrics_cumsum['MaxAE'],
            'SMAPE_cumsum': fc_metrics_cumsum['SMAPE'],
            'min_fact': np.min(work_days_only['t']),
            'max_fact': np.min(work_days_only['t']),
            'mean_fact': np.mean(work_days_only['t']),
            'std_fact': np.std(work_days_only['t'], ddof=1),
            'sum_fact': np.sum(work_days_only['t']),
            'P90_fact': np.percentile(work_days_only['t'], 90),
            'P95_fact': np.percentile(work_days_only['t'], 95)
        }, index=y_fc_i.index)
        return result
