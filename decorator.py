import os
import re

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression, mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.stattools import pacf

from describer import Describer
from utils import symmetric_mean_absolute_percentage_error, plot_go, max_absolute_error


class Decorator:
    def __init__(self, weekends: pd.DataFrame):
        """
        Constructor for the Decorator object

        :param weekends: pd.DataFrame of weekends and work days
        """

        self.weekends = weekends
        self.lags = None
        self.models_selection = False

    @staticmethod
    def __sort_columns(columns: list) -> list:
        """
        Sorting list of columns alphabetically with numbers

        :param columns: list of unsorted columns
        :return: list of sorted columns
        """
        def generate_keys(strings: list):
            return [int(key) if key.isdigit() else key for key in re.split(r'(\d+)', strings)]

        return sorted(columns, key=generate_keys)

    @staticmethod
    def __feature_mean(data, cat_feature, real_feature):
        return dict(data.groupby(cat_feature)[real_feature].mean())

    @staticmethod
    def __feature_sum(data, cat_feature, real_feature):
        return dict(data.groupby(cat_feature)[real_feature].sum())

    @staticmethod
    def calculate_metrics(actual: np.ndarray, prediction: np.ndarray, plot_name: str, silent: bool = None,
                          plot_path: str = None, csv_path: str = None, result_path: str = None) -> dict:
        """
        Calculating metrics between actual (y_true) and prediction (y_pred). Also provide functions to draw/save plots
        and save metrics with pd.DataFrame of actual, prediction and residuals values

        :param actual: np.ndarray holding the y_true values
        :param prediction: np.ndarray holding the y_pred values
        :param plot_name: str of names for plotting and col indices in metrics
        :param silent: if True show the plots
        :param plot_path: str of path for saving plots
        :param csv_path: str of path for saving metrics
        :param result_path: str of path for saving actual, prediction and residuals values
        :return: dict of metrics
        """
        assert isinstance(actual, np.ndarray), f'[TypeError] Actual must be in {np.ndarray} class'
        assert isinstance(prediction, np.ndarray), f'[TypeError] Prediction must be in {np.ndarray} class'

        result = pd.DataFrame({
            'actual': actual,
            'prediction': prediction
        }).fillna(0)
        result['diff'] = result['actual'] - result['prediction']

        Describer.hist(result[['diff']], path=f'{plot_path}{plot_name}')
        Describer.boxplot(result[['diff']], path=f'{plot_path}{plot_name}')
        Describer.scatter(result[['diff']], path=f'{plot_path}{plot_name}')

        metrics = {
            'MAE': mean_absolute_error(result['actual'].values, result['prediction'].values),
            'MAPE': mean_absolute_percentage_error(result['actual'].values, result['prediction'].values),
            'MaxAE': max_absolute_error(result['actual'].values, result['prediction'].values),
            'SMAPE': symmetric_mean_absolute_percentage_error(result['actual'].values, result['prediction'].values)
        }
        stats = {
            'min_fc': np.min(result['prediction']),
            'max_fc': np.max(result['prediction']),
            'mean_fc': np.mean(result['prediction']),
            'std_fc': np.std(result['prediction'], ddof=1),
            'sum_fc': np.sum(result['prediction']),
            'P90_fc': np.percentile(result['prediction'], 90),
            'P95_fc': np.percentile(result['prediction'], 95)
        }

        metrics.update(stats)

        print(f'\n\033[4m{"Model": <50}{"MAE": >28}{"MAPE": >16}{"SMAPE": >10}{"": <1}\033[0m')
        print(f'{plot_name:<50}{metrics["MAE"]:>36.3f},{metrics["MAPE"]:>8.3f},{metrics["SMAPE"]:>6.3f}')
        plot_go(traces=[[result['actual'].index, result['actual'], 'Base'],
                        [result['actual'].index, result['prediction'], 'Prediction'],
                        [result['actual'].index, result['diff'], 'Diff']],
                title=f'{plot_name} Test Prediction. Mean diff: {result["diff"].mean()}',
                errors=[round(metrics['MAE'], 3), round(metrics['MAPE'], 3), round(metrics['SMAPE'], 3)],
                save_path=f'{plot_path}{plot_name}.png', silent=silent)
        if result_path:
            result.to_csv(result_path)
        if csv_path:
            if os.path.isfile(csv_path):
                metrics_df = pd.read_csv(csv_path, index_col=0)
            else:
                metrics_df = pd.DataFrame()
            metrics_df[plot_name] = pd.DataFrame().from_dict(metrics, orient='index').iloc[:, 0]
            metrics_df.to_csv(csv_path)
        return metrics

    @staticmethod
    def feature_selection(features: pd.DataFrame, target: pd.DataFrame, max_features: int) -> list:
        """
        Selecting features using Lasso, F-regression, Mutual info regression models

        :param features: pd.DataFrame holding the input time-series features
        :param target: pd.DataFrame holding the input time-series target
        :param max_features: number of maximum features for selection in each model
        :return: list of selected features
        """
        model = linear_model.Lasso(alpha=0.05).fit(features, target)
        selector = SelectFromModel(model, prefit=True, max_features=max_features)
        selected_lasso = features.columns[(selector.get_support())]

        selector = SelectKBest(f_regression, k=max_features).fit(features, target['t'].ravel())
        selector.get_support(indices=True)
        selected_f_regression = features.columns[(selector.get_support())]

        selector = SelectKBest(mutual_info_regression, k=max_features).fit(features, target['t'].ravel())
        selector.get_support(indices=True)
        selected_mutual_regression = features.columns[(selector.get_support())]

        union_selected = selected_lasso.union(selected_f_regression).union(selected_mutual_regression)
        print(f'Total features: {features.shape[1]}')
        print(f'Selected features from Lasso: {len(selected_lasso)}')
        print(f'Selected features from F-regression: {len(selected_f_regression)}')
        print(f'Selected features from Mutual info regression: {len(selected_mutual_regression)}')
        print(f'Union features: {len(union_selected)}')
        print(union_selected)
        return union_selected

    def generate_lag_features(self, df: pd.DataFrame, n_lags: int, threshold: float = 0.2) -> pd.DataFrame:
        """
        Generating lag features using the input time series target

        :param df: pd.DataFrame holding the input time-series target
        :param n_lags: number of lags in the forecasting features
        :param threshold: number of partial autocorrelation function, equivalent to 5% relevance for the lag
        :return: pd.DataFrame with generated lag features and list of lags
        """
        if self.lags is None:
            partial = pd.Series(data=pacf(df['t'], nlags=n_lags))
            self.lags = list(partial[np.abs(partial) >= threshold].index)
            if 0 in self.lags:
                self.lags.remove(0)
        if len(self.lags) == 0:
            self.lags = list(range(1, n_lags))
            self.models_selection = True
        # t-lags
        df = pd.concat([df, pd.concat([df[['t']].shift(i).add_suffix(f'_{i}') for i in self.lags], axis=1)], axis=1)

        # sum over 5 days (work week)
        df = pd.concat([df, df[f't_{self.lags[0]}'].rolling(window=5).sum().rename('5d_week_sum')], axis=1)

        # lags over work weeks
        df = pd.concat([df, pd.concat([df[['5d_week_sum']].shift(i).add_suffix(f'_{i}') for i in self.lags], axis=1)],
                       axis=1)

        # rolling days mean over first t-lag
        rolling_days = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
        df = pd.concat([
            df, pd.concat([df[f't_{self.lags[0]}'].rolling(window=i).mean().rename(f'mean_{i}') for i in rolling_days],
                          axis=1)
        ], axis=1)

        # expanding mean/sum over first t-lag
        df = pd.concat([df, df[f't_{self.lags[0]}'].expanding(2).mean().rename('expending_mean')], axis=1)
        df = pd.concat([df, df[f't_{self.lags[0]}'].expanding(2).sum().rename('expending_sum')], axis=1)
        pct_d = df[f't_{self.lags[0]}'].pct_change(freq='D')
        pct_d[pct_d.isin([np.inf, np.nan, -np.inf])] = 0
        df = pd.concat([df, pct_d.rename('pct_d')], axis=1)

        # ms_resample = df[f't_{self.lags[0]}'].resample('MS').asfreq().dropna()
        # ms_resample[ms_resample.isin([np.inf, -np.inf])] = 0
        # df = pd.concat([df, ms_resample.pct_change().resample('D').pad().rename('pct_month_mean')], axis=1)
        # qs_resample = df[f't_{self.lags[0]}'].resample('Q').asfreq().dropna()
        # qs_resample[qs_resample.isin([np.inf, -np.inf])] = 0
        # df = pd.concat([df, qs_resample.pct_change().resample('D').bfill().rename('pct_quarter_mean')], axis=1)

        if self.models_selection:
            self.lags = None

        df.dropna(inplace=True)
        return df

    def generate_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generating calendar features using the input time series target

        :param df: pd.DataFrame holding the input time-series target
        :return: pd.DataFrame with generated calendar features
        """
        #############################################################
        # ################## TEMPORARY FEATURES #####################
        #############################################################
        df = pd.concat([df, pd.Series(df.index.weekday, index=df.index).rename('weekday')], axis=1)
        df = pd.concat([df, pd.Series(df.index.isocalendar().week, index=df.index).rename('week')], axis=1)
        df = pd.concat([df, pd.Series(df.index.month, index=df.index).rename('month')], axis=1)
        df = pd.concat([df, pd.Series(df.index.quarter, index=df.index).rename('quarter')], axis=1)

        #############################################################
        # ################### CALENDAR FEATURES #####################
        #############################################################
        weekends = self.weekends.loc[df.index[0]: df.index[-1]]
        df = pd.concat([df, weekends['flag'].rename('is_weekend')], axis=1)
        df = pd.concat([df, pd.Series(df.index.is_month_start * 1, index=df.index).rename('is_month_start')], axis=1)
        df = pd.concat([df, pd.Series(df.index.is_month_end * 1, index=df.index).rename('is_month_end')], axis=1)
        df = pd.concat([df, pd.Series(df.index.is_quarter_start * 1, index=df.index).rename('is_quarter_start')],
                       axis=1)
        df = pd.concat([df, pd.Series(df.index.is_quarter_end * 1, index=df.index).rename('is_quarter_end')], axis=1)
        df = pd.concat([df, pd.Series(df.index.is_year_start * 1, index=df.index).rename('is_year_start')], axis=1)
        df = pd.concat([df, pd.Series(df.index.is_year_end * 1, index=df.index).rename('is_year_end')], axis=1)

        lowest_lag = 't' if self.lags is None else f't_{self.lags[0]}'

        # Mean calendar features
        df = pd.concat([df, pd.Series(list(map(self.__feature_mean(df, 'weekday', lowest_lag).get, df.weekday)),
                                      index=df.index).rename('weekday_mean')], axis=1)
        df = pd.concat([df, pd.Series(list(map(self.__feature_mean(df, 'week', lowest_lag).get, df.week)),
                                      index=df.index).rename('week_mean')], axis=1)
        df = pd.concat([df, pd.Series(list(map(self.__feature_mean(df, 'month', lowest_lag).get, df.month)),
                                      index=df.index).rename('month_mean')], axis=1)
        df = pd.concat([df, pd.Series(list(map(self.__feature_mean(df, 'quarter', lowest_lag).get, df.quarter)),
                                      index=df.index).rename('quarter_mean')], axis=1)
        df = pd.concat([df, pd.Series(df[lowest_lag].rolling(window=180).mean(), index=df.index).rename('half_mean')],
                       axis=1)
        df = pd.concat([df, pd.Series(df[lowest_lag].rolling(window=365).mean(), index=df.index).rename('year_mean')],
                       axis=1)

        # Sum calendar features
        df = pd.concat([df, pd.Series(list(map(self.__feature_sum(df, 'weekday', lowest_lag).get, df.weekday)),
                                      index=df.index).rename('weekday_sum')], axis=1)
        df = pd.concat([df, pd.Series(list(map(self.__feature_sum(df, 'week', lowest_lag).get, df.week)),
                                      index=df.index).rename('7d_week_sum')], axis=1)
        df = pd.concat([df, pd.Series(list(map(self.__feature_sum(df, 'month', lowest_lag).get, df.month)),
                                      index=df.index).rename('month_sum')], axis=1)
        df = pd.concat([df, pd.Series(list(map(self.__feature_sum(df, 'quarter', lowest_lag).get, df.quarter)),
                                      index=df.index).rename('quarter_sum')], axis=1)
        df = pd.concat([df, pd.Series(df[lowest_lag].rolling(window=180).sum(), index=df.index).rename('half_sum')],
                       axis=1)
        df = pd.concat([df, pd.Series(df[lowest_lag].rolling(window=365).sum(), index=df.index).rename('year_sum')],
                       axis=1)

        df.drop(['weekday', 'week', 'month', 'quarter'], axis=1, inplace=True)
        df.dropna(inplace=True)
        return df

    def generate_features(self, df: pd.DataFrame, n_lags: int, threshold: float = 0.2) -> (pd.DataFrame, pd.DataFrame):
        """
        Generating features using the input time series target

        :param df: pd.DataFrame holding the input time-series target
        :param n_lags: number of lags in the forecasting features
        :param threshold: number of partial autocorrelation function, equivalent to 5% relevance for the lag
        :return: pd.DataFrame with generated features and pd.DataFrame with target
        :return: pd.DataFrame with target
        """

        def generate_step(_df: pd.DataFrame):
            _features = pd.DataFrame(_df.copy())
            _features.columns = ["t"]

            _features = self.generate_lag_features(df=_features, n_lags=n_lags, threshold=threshold)
            _features = self.generate_calendar_features(df=_features)

            # ##########################################################################################################
            # NUM WORKDAY IN MONTH
            # count = _features.groupby(pd.Grouper(freq='M')).sum().shape[0]
            # start_date, end_date = pd.Timestamp('2015-01-10').date(), pd.Timestamp('2015-02-10').date()
            # num_workday = pd.Series(name='num_workday')
            # while count > 0:
            #     num_workday_month_i = pd.Series(_features.loc[_features['is_weekend'] == 0].loc[
            #                                     start_date + pd.Timedelta(days=1):end_date].reset_index().index + 1,
            #                                     index=_features.loc[_features['is_weekend'] == 0].loc[
            #                                     start_date + pd.Timedelta(days=1):end_date].index)
            #     num_workday = pd.concat([num_workday, num_workday_month_i.rename('num_workday')], axis=0)
            #     start_date = end_date
            #     end_date = (start_date + pd.DateOffset(months=1)).date()
            #     count -= 1
            # _features = pd.concat([_features, num_workday], axis=1).fillna(0)
            # _features['num_workday'] = _features['num_workday'].astype(int)
            # ##########################################################################################################

            _target = pd.DataFrame(_features.loc[:, 't'].copy())
            _features.drop(['t'], axis=1, inplace=True)

            return _features, _target

        features, target = generate_step(df)

        if self.models_selection:
            # Feature Selection based on Lasso, F-regression, Mutual info regression
            selected_features = self.feature_selection(features=features, target=target, max_features=30)
            if self.lags is None:
                self.lags = sorted(
                    [int(i.split('_')[1]) for i in selected_features if 't' in i.split('_') and len(i.split('_')) == 2])
                self.models_selection = False
                # Generate df with selected lags again
                features, target = generate_step(df)

        # features = features.fillna(0)
        return features, target
