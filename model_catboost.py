import os

import numpy as np
import pandas as pd
import wandb

from decorator import Decorator
from describer import Describer
from forecaster import Forecaster
from models.catboost import CatboostModel
from sql_scripts import save_sql_cvr
from transformer import Transformer
from utils import plot_go


class Model:
    def __init__(self, df: pd.DataFrame, plan: pd.Series, weekends: pd.DataFrame, params: dict):
        """
        Constructor for the Model object

        :param df: pd.DataFrame holding the input time-series target
        :param plan: pd.Series holding the planned sum over year
        :param weekends: pd.DataFrame of weekends and work days
        :param params: dict of any params
        """

        self.seed = params['seed']
        self.params = params
        self.plan = plan
        self.weekends = weekends

        self.decorator = Decorator(weekends=self.weekends)
        self.transformer = Transformer(scale=True, diff=False, log=False, detrend=False)
        self.target = self.transformer.transform(df[['first_difference']])
        Describer.describe(self.target)

    def forecast_month(self):
        """
        Forecasting over months from the 10th day of current month to the 10th day of next month
        """

        start_date, end_date = self.params["start_date"], self.params["end_date"]
        # predict over 10 month from 2021-01-10
        count = 10
        while count != 0:
            period = f'{start_date}_{end_date}'
            print(f"[INFO] ID: {self.params['id']} forecast {period}")

            wandb.init(
                project="aisa-group",
                tags=["cvr", period, "diff"],
                group=f"cvr:{self.params['id']}:diff",
                notes="one month forecast",
                job_type=period)

            num_prediction = (end_date - start_date).days

            # Choose {num_prediction} time periods (~days) as forecasting horizon
            features, target = self.decorator.generate_features(df=self.target.loc[:end_date],
                                                                n_lags=num_prediction)

            X_train_i, y_train_i = features.loc[:start_date], target.loc[:start_date]
            X_test_i = features.loc[start_date + pd.Timedelta(days=1):end_date]
            y_test_i = target.loc[start_date + pd.Timedelta(days=1):end_date]

            model_i, cv_score, test_score = CatboostModel.run(X_train_i, y_train_i, max_evals=10)
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

            self.decorator.calculate_metrics(actual=y_test_i.loc[work_days_ind.index]['t'].values,
                                             prediction=y_pred_i.loc[work_days_ind.index]['t'].values,
                                             plot_name=model_name + "_pred", silent=True,
                                             plot_path=self.params['path_plots'], csv_path=csv_path)

            csv_path = f"{self.params['path_metrics']}metrics_fc.csv"
            result_path = f"{self.params['path_metrics']}{period}.csv"
            fc_metrics_diff = self.decorator.calculate_metrics(actual=y_test_i.loc[work_days_ind.index]['t'].values,
                                                               prediction=y_fc_i.loc[work_days_ind.index]['t'].values,
                                                               plot_name=model_name + "_fc_diff", silent=True,
                                                               plot_path=self.params['path_plots'], csv_path=csv_path,
                                                               result_path=result_path)
            y_test_i_cumsum = y_test_i.loc[work_days_ind.index].groupby(pd.Grouper(freq='Y')).cumsum()
            y_fc_i_cumsum = y_fc_i.loc[work_days_ind.index].groupby(pd.Grouper(freq='Y')).cumsum()
            fc_metrics_cumsum = self.decorator.calculate_metrics(actual=y_test_i_cumsum.loc[work_days_ind.index]['t'].values,
                                                                 prediction=y_fc_i_cumsum.loc[work_days_ind.index]['t'].values,
                                                                 plot_name=model_name + "_fc_cumsum", silent=True,
                                                                 plot_path=self.params['path_plots'], csv_path=csv_path,
                                                                 result_path=result_path)

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
                'min_fact':  np.min(work_days_only['t']),
                'max_fact': np.min(work_days_only['t']),
                'mean_fact': np.mean(work_days_only['t']),
                'std_fact': np.std(work_days_only['t'], ddof=1),
                'sum_fact': np.sum(work_days_only['t']),
                'P90_fact': np.percentile(work_days_only['t'], 90),
                'P95_fact': np.percentile(work_days_only['t'], 95)
            }, index=y_fc_i.index)
            # save_sql_cvr(result, self.params['model_ver'])

            params_wandb = {
                'cumsum': False,
                'diff': self.transformer.diff,
                'scale': self.transformer.scale,
                'detrend': self.transformer.detrend,
                'base_data_type': self.target.columns[0],
                'model_id': self.params["id"],
            }

            wandb.log({f'{k}_diff': v for k, v in fc_metrics_diff.items()})
            wandb.log({f'{k}_cumsum': v for k, v in fc_metrics_cumsum.items()})
            wandb.log(params_wandb)
            wandb.finish()

            start_date = (start_date + pd.DateOffset(months=1)).date()
            end_date = (start_date + pd.DateOffset(months=1)).date()
            count -= 1
