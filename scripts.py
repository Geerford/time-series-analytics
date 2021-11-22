import os

import pandas as pd
from sklearn import linear_model
from sqlalchemy import create_engine
from statsmodels.tsa.seasonal import seasonal_decompose

import model
from sql_scripts import read_sql_cvr, read_sql_weekend, save_sql_cvr
from utils import load_pkl, save_model, time_series_thresholds, save_yaml


# noinspection DuplicatedCode
def predict_cvr_threshold(state: str, model_name: str, model_ver: str, data_from: str, num_exp: int, horizon: str,
                          season_decompose: bool):
    """
    Predict CVR with threshold

    :param state: state of model (train/test)
    :param model_name: name of model algorithm (catboost, sarimax, etc)
    :param model_ver: version of experiment to save into db
    :param data_from: name of data source (sql/csv)
    :param num_exp: number of experiment to local save
    :param horizon: horizon of forecasting
    :param season_decompose: seasonal decomposition of time series
    """
    mdlParams = {
        'seed': 21,
        'plotForecast': True,
        'plotLoss': True,
        'plotPrediction': True,
        'plotTS': True,
        'silentPlot': True,
        'period': horizon
    }
    scrParams = {
        'model_ver': model_ver + str(num_exp),
        'path_data': 'data/',
        'path_exp': f'experiments/exp/cvr/exp_{num_exp}/',
        'csv_name': 'cvr.csv',
        'cvr_codes_21': 'cvr_codes_21.csv',
        'weekends_path': 'data/weekends.csv',
        'skipped_codes_pkl': 'skipped_codes.pkl',
        'state': state,
        'model_name': model_name,
        'start_date': pd.Timestamp('2021-01-10').date(),
        'end_date': pd.Timestamp('2021-02-10').date(),
        'forecast_count': 10,
        'freq': mdlParams['period'],
        'aggregate': False,
        'input_codes': [610],
        'season_decompose': season_decompose
    }
    os.makedirs('config', exist_ok=True)
    save_yaml(mdlParams, 'model_config')
    save_yaml(scrParams, 'script_config')

    df_cvr = read_sql_cvr() if data_from == 'sql' else pd.read_csv(f'{scrParams["path_data"]}{scrParams["csv_name"]}',
                                                                   index_col='docdate', parse_dates=['docdate'],
                                                                   infer_datetime_format=True)
    weekends = read_sql_weekend() if data_from == 'sql' else pd.read_csv(scrParams['weekends_path'],
                                                                         index_col='docdate', parse_dates=['docdate'],
                                                                         infer_datetime_format=True)
    cvr_codes = pd.read_csv(f"{scrParams['path_data']}{scrParams['cvr_codes_21']}",
                            header=None, dtype=int).iloc[:, 0].values

    os.makedirs(scrParams["path_exp"], exist_ok=True)

    if scrParams['aggregate']:
        # Aggregate over all available codes
        df_groupped_cvr = df_cvr.loc[df_cvr['cvr'].isin(cvr_codes)]
        del df_cvr
        df_aggregated = df_groupped_cvr.loc[:, ['sum_ex', 'sum_pl']].groupby(df_groupped_cvr.index).agg('sum')

        threshold = time_series_thresholds(df_aggregated['sum_ex'])
        for std in threshold:
            df_threshold = df_aggregated.copy()
            if std is None:
                step(df=df_aggregated, weekends=weekends, model_params=mdlParams, script_params=scrParams)
            else:
                df_threshold.loc[df_threshold['sum_ex'] > std, 'sum_ex'] = std
                step(df=df_threshold, weekends=weekends, model_params=mdlParams, script_params=scrParams)
    else:
        # Iterate over input codes
        if not scrParams['input_codes']:
            # If input codes is empty, add all codes
            scrParams['input_codes'] = cvr_codes
        df_groupped_cvr = [group for _, group in df_cvr.groupby(['cvr']) if group['cvr'][0] in scrParams['input_codes']]
        del df_cvr

        for _, group_cvr in enumerate(df_groupped_cvr):
            threshold = time_series_thresholds(group_cvr['sum_ex'])
            for std in threshold:
                df_threshold = group_cvr.copy()
                if std is None:
                    step(df=group_cvr, weekends=weekends, model_params=mdlParams, script_params=scrParams)
                else:
                    df_threshold.loc[df_threshold['sum_ex'] > std, 'sum_ex'] = std
                    step(df=df_threshold, weekends=weekends, model_params=mdlParams, script_params=scrParams)


# noinspection DuplicatedCode
def predict_cvr(state: str, model_name: str, model_ver: str, data_from: str, num_exp: int, horizon: str,
                season_decompose: bool):
    """
    Predict CVR

    :param state: state of model (train/test)
    :param model_name: name of model algorithm (catboost, sarimax, etc)
    :param model_ver: version of experiment to save into db
    :param data_from: name of data source (sql/csv)
    :param num_exp: number of experiment to local save
    :param horizon: horizon of forecasting
    :param season_decompose: seasonal decomposition of time series
    """
    mdlParams = {
        'seed': 21,
        'plotForecast': True,
        'plotLoss': True,
        'plotPrediction': True,
        'plotTS': True,
        'silentPlot': True,
        'freq': horizon
    }
    scrParams = {
        'model_ver': model_ver + str(num_exp),
        'path_data': 'data/',
        'path_exp': f'experiments/exp/cvr/exp_{num_exp}/',
        'csv_name': 'cvr.csv',
        'cvr_codes_21': 'cvr_codes_21.csv',
        'weekends_path': 'data/weekends.csv',
        'skipped_codes_pkl': 'skipped_codes.pkl',
        'state': state,
        'model_name': model_name,
        'start_date': pd.Timestamp('2021-01-10').date(),
        'end_date': pd.Timestamp('2021-02-10').date(),
        'forecast_count': 10,
        'aggregate': False,
        'input_codes': [610],
        'season_decompose': season_decompose
    }
    os.makedirs('config', exist_ok=True)
    save_yaml(mdlParams, 'model_config')
    save_yaml(scrParams, 'script_config')

    df_cvr = read_sql_cvr() if data_from == 'sql' else pd.read_csv(f'{scrParams["path_data"]}{scrParams["csv_name"]}',
                                                                   index_col='docdate', parse_dates=['docdate'],
                                                                   infer_datetime_format=True)
    weekends = read_sql_weekend() if data_from == 'sql' else pd.read_csv(scrParams['weekends_path'],
                                                                         index_col='docdate', parse_dates=['docdate'],
                                                                         infer_datetime_format=True)
    cvr_codes = pd.read_csv(f"{scrParams['path_data']}{scrParams['cvr_codes_21']}",
                            header=None, dtype=int).iloc[:, 0].values

    os.makedirs(scrParams["path_exp"], exist_ok=True)

    if scrParams['aggregate']:
        # Aggregate over all available codes
        df_groupped_cvr = df_cvr.loc[df_cvr['cvr'].isin(cvr_codes)]
        del df_cvr
        df_aggregated = df_groupped_cvr.loc[:, ['sum_ex', 'sum_pl']].groupby(df_groupped_cvr.index).agg('sum')
        step(df=df_aggregated, weekends=weekends, model_params=mdlParams, script_params=scrParams)
    else:
        # Iterate over input codes
        if not scrParams['input_codes']:
            # If input codes is empty, add all codes
            scrParams['input_codes'] = cvr_codes
        df_groupped_cvr = [group for _, group in df_cvr.groupby(['cvr']) if group['cvr'][0] in scrParams['input_codes']]
        del df_cvr

        for _, group_cvr in enumerate(df_groupped_cvr):
            step(df=group_cvr, weekends=weekends, model_params=mdlParams, script_params=scrParams)


def step(df: pd.DataFrame, weekends: pd.DataFrame, model_params: dict, script_params: dict):
    """
    Step of grouping time series

    :param df: pd.DataFrame holding the input time-series target
    :param weekends: pd.DataFrame of weekends and work days
    :param model_params: dict of model parameters
    :param script_params: dict of script parameters
    """
    plan = df['sum_pl'].groupby(pd.Grouper(freq='Y')).agg('sum')
    df.drop(['sum_pl'], axis=1, inplace=True)

    cvr = pd.DataFrame(df['sum_ex'].resample('D').asfreq().fillna(0))

    cvr.columns = ['first_difference']
    cvr['cumsum'] = cvr['first_difference'].groupby(pd.Grouper(freq='Y')).cumsum()
    cvr['second_difference'] = cvr['first_difference'].groupby(pd.Grouper(freq='Y')).diff().fillna(0)

    cvr_id = df['cvr'][0]
    model_params['path_models'] = f'{script_params["path_exp"]}{cvr_id}/models/'
    model_params['path_metrics'] = f'{script_params["path_exp"]}{cvr_id}/csv/'
    model_params['path_plots'] = f'{script_params["path_exp"]}{cvr_id}/plots/'
    model_params['id'] = cvr_id

    skipped_codes = load_pkl(f'{script_params["path_exp"]}{script_params["skipped_codes_pkl"]}')

    cvr_month = pd.DataFrame()
    if model_params['freq'] == 'monthly':
        # Sum over 11 of current month to 10 of next month
        start_date, end_date = pd.Timestamp('2014-01-10').date(), pd.Timestamp('2014-02-10').date()
        count = cvr[['first_difference']].groupby(pd.Grouper(freq='M')).sum().loc['2014-01-10':].shape[0]
        while count > 0:
            cvr_true_month_i = cvr[['first_difference']].loc[start_date + pd.Timedelta(days=1):end_date]
            cvr_true_month_sum = pd.Series(cvr_true_month_i.sum()['first_difference'], index=[start_date])
            cvr_month = pd.concat([cvr_month, cvr_true_month_sum], axis=0)
            start_date = end_date
            end_date = (start_date + pd.DateOffset(months=1)).date()
            count -= 1
        cvr_month.columns = ['first_difference']
        cvr_month.index = pd.Series(cvr_month.index).apply(lambda x: pd.Timestamp(x))
        if cvr_month.shape[0] < 48:
            skipped_codes.append({'cvr': cvr_id})
            save_model(f'{script_params["path_exp"]}{script_params["skipped_codes_pkl"]}', skipped_codes)
            print(f'[SKIP] Skipped code: {cvr_id}')
            return
    else:
        if cvr.shape[0] < 365:
            skipped_codes.append({'cvr': cvr_id})
            save_model(f'{script_params["path_exp"]}{script_params["skipped_codes_pkl"]}', skipped_codes)
            print(f'[SKIP] Skipped code: {cvr_id}')
            return

    if script_params['state'] == 'load':
        pass
    else:
        df = cvr_month if model_params['freq'] == 'monthly' else cvr

        # Predict trend used seasonal decomposition
        if script_params['season_decompose']:
            period = 12
            df_decomposed = seasonal_decompose(df[['first_difference']], model='additive', period=period)

            preds = []
            trend = df_decomposed.trend.dropna().copy()[:-4]
            trend.index = pd.Series(trend.index).apply(lambda x: pd.Timestamp(x))
            month_count, ind_count = script_params['forecast_count'], 0
            while month_count != 0:
                last_forecast = preds[-1] if len(preds) > 0 else trend[-1]
                ind = [cvr['first_difference'][-script_params['forecast_count']:].index.values[ind_count]]
                trend = trend.append(pd.Series(last_forecast, index=ind))

                model_i = linear_model.LinearRegression()
                model_i.fit(trend.index.map(pd.Timestamp.toordinal).values.reshape(-1, 1), trend.values)

                preds.append(model_i.predict(trend.index.map(pd.Timestamp.toordinal).values.reshape(-1, 1))[-1])
                month_count -= 1
                ind_count += 1

            season = df_decomposed.seasonal[-(script_params['forecast_count'] + period):
                                            -script_params['forecast_count']].iloc[:len(preds)].values
            fc = pd.Series(preds + season, index=trend[-script_params['forecast_count']:].index)
            pd.DataFrame(
                {
                    'fact': df_decomposed.observed,
                    'trend': df_decomposed.trend,
                    'seasonal': df_decomposed.seasonal,
                    'trend_fc': pd.Series(preds, index=trend[-script_params['forecast_count']:].index),
                    'season_fc': pd.Series(season, index=trend[-script_params['forecast_count']:].index),
                    'forecast': fc
                }).to_csv('seasonal.csv')
            return

        _model = getattr(model, 'Model')
        model_cvr = _model(df, plan, weekends, model_params, script_params['model_name'])

        start_date, end_date = script_params['start_date'], script_params['end_date']
        # Predict over {forecast_count} month from {start_date}
        count = script_params['forecast_count']
        while count != 0:
            num_prediction = 1 if model_params['freq'] == 'monthly' else (end_date - start_date).days
            result = model_cvr.forecast(start_date, end_date, num_prediction)
            # save_sql_cvr(result, scriptParams['model_ver'])
            start_date = end_date
            end_date = (start_date + pd.DateOffset(months=1)).date()
            count -= 1
