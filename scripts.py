import os

import pandas as pd
from sklearn import linear_model
from sqlalchemy import create_engine
from statsmodels.tsa.seasonal import seasonal_decompose

import model
from sql_scripts import read_sql_cvr, read_sql_weekend, save_sql_cvr
from utils import load_pkl, save_model


# noinspection DuplicatedCode
def predict_cvr_daily_anomaly(state: str, model_name: str, model_ver: str, data_from: str, force: bool, num_exp: int):
    mdlParams = {
        'path_data': 'data/',
        'path_skipped': f'experiments/exp/cvr/exp_{num_exp}/',
        'seed': 21,
        'plotForecast': True,
        'plotLoss': True,
        'plotPrediction': True,
        'plotTS': True,
        'silentPlot': True,
        'model_ver': model_ver,
        'period': 'daily'
    }
    csv_name, cvr_codes_21, weekends_path = 'cvr.csv', 'cvr_codes_21.csv', f'{mdlParams["path_data"]}weekends.csv'

    mdlParams['start_date'], mdlParams['end_date'] = pd.Timestamp('2021-01-10').date(), pd.Timestamp('2021-02-10').date()

    df_cvr = read_sql_cvr() if data_from == 'sql' else pd.read_csv(f'{mdlParams["path_data"]}{csv_name}',
                                                                   index_col='docdate', parse_dates=['docdate'],
                                                                   infer_datetime_format=True)
    weekends = read_sql_weekend() if data_from == 'sql' else pd.read_csv(weekends_path, index_col='docdate',
                                                                         parse_dates=['docdate'],
                                                                         infer_datetime_format=True)

    cvr_codes = pd.read_csv(f"{mdlParams['path_data']}{cvr_codes_21}", header=None, dtype=int).iloc[:, 0].values

    # TODO aggregate over cvr
    aggregate = False
    if aggregate:
        df_groupped_cvr = df_cvr  # .loc[~df_cvr['cvr'].isin([110, 120])]
        group_cvr = df_groupped_cvr.loc[:, ['sum_ex', 'sum_pl']].groupby(df_groupped_cvr.index).agg('sum')

    input_cvr_codes = [610]
    if not input_cvr_codes:
        # If input codes is empty, add all codes
        input_cvr_codes = cvr_codes

    df_groupped_cvr = [group for _, group in df_cvr.groupby(['cvr']) if group['cvr'][0] in input_cvr_codes]

    del df_cvr
    for ind, group_cvr in enumerate(df_groupped_cvr):
        cvr_plan = group_cvr['sum_pl'].groupby(pd.Grouper(freq='Y')).agg('sum')
        group_cvr.drop(['sum_pl'], axis=1, inplace=True)
        cvr_id = group_cvr['cvr'][0]

        import numpy as np
        threshold = [3 * np.std(group_cvr['sum_ex'], ddof=1),
                     4 * np.std(group_cvr['sum_ex'], ddof=1),
                     5 * np.std(group_cvr['sum_ex'], ddof=1),
                     np.percentile(group_cvr['sum_ex'], 90),
                     np.percentile(group_cvr['sum_ex'], 95),
                     None]
        for std in threshold:
            group_cvr_todo = group_cvr.copy()

            if std is None:
                cvr = pd.DataFrame(group_cvr['sum_ex'].resample('D').asfreq().fillna(0))
            else:
                group_cvr_todo.loc[group_cvr_todo['sum_ex'] > std, 'sum_ex'] = std
                cvr = pd.DataFrame(group_cvr_todo['sum_ex'].resample('D').asfreq().fillna(0))

            cvr.columns = ['first_difference']
            cvr['cumsum'] = cvr['first_difference'].groupby(pd.Grouper(freq='Y')).cumsum()
            cvr['second_difference'] = cvr['first_difference'].groupby(pd.Grouper(freq='Y')).diff().fillna(0)

            # cvr_id = 'agg'
            mdlParams['path_models'] = f'experiments/exp/cvr/exp_{num_exp}/{cvr_id}/models/'
            mdlParams['path_metrics'] = f'experiments/exp/cvr/exp_{num_exp}/{cvr_id}/csv/'
            mdlParams['path_plots'] = f'experiments/exp/cvr/exp_{num_exp}/{cvr_id}/plots/'

            if state == 'load':
                pass
            else:
                num_exp += 1
                mdlParams['id'] = cvr_id

                _model = getattr(model, 'Model')
                model_cvr = _model(cvr, cvr_plan, weekends, mdlParams, model_name)
                result = model_cvr.forecast()
                # save_sql_cvr(result, mdlParams['model_ver'])
    return True


# noinspection DuplicatedCode
def predict_cvr_daily(state: str, model_name: str, model_ver: str, data_from: str, force: bool, num_exp: int):
    mdlParams = {
        'path_data': 'data/',
        'path_skipped': f'experiments/exp/cvr/exp_{num_exp}/',
        'seed': 21,
        'plotForecast': True,
        'plotLoss': True,
        'plotPrediction': True,
        'plotTS': True,
        'silentPlot': True,
        'model_ver': model_ver,
        'period': 'daily'
    }
    csv_name, cvr_codes_21, weekends_path = 'cvr.csv', 'cvr_codes_21.csv', f'{mdlParams["path_data"]}weekends.csv'
    skipped_codes_pkl = 'skipped_codes.pkl'

    df_cvr = read_sql_cvr() if data_from == 'sql' else pd.read_csv(f'{mdlParams["path_data"]}{csv_name}',
                                                                   index_col='docdate', parse_dates=['docdate'],
                                                                   infer_datetime_format=True)
    weekends = read_sql_weekend() if data_from == 'sql' else pd.read_csv(weekends_path, index_col='docdate',
                                                                         parse_dates=['docdate'],
                                                                         infer_datetime_format=True)
    cvr_codes = pd.read_csv(f"{mdlParams['path_data']}{cvr_codes_21}", header=None, dtype=int).iloc[:, 0].values

    # TODO aggregate over cvr
    aggregate = False
    if aggregate:
        df_groupped_cvr = df_cvr  # .loc[~df_cvr['cvr'].isin([110, 120])]
        group_cvr = df_groupped_cvr.loc[:, ['sum_ex', 'sum_pl']].groupby(df_groupped_cvr.index).agg('sum')

    input_cvr_codes = [610]
    if not input_cvr_codes:
        # If input codes is empty, add all codes
        input_cvr_codes = cvr_codes

    df_groupped_cvr = [group for _, group in df_cvr.groupby(['cvr']) if group['cvr'][0] in input_cvr_codes]

    del df_cvr
    os.makedirs(mdlParams["path_skipped"], exist_ok=True)
    skipped_codes = load_pkl(f'{mdlParams["path_skipped"]}{skipped_codes_pkl}')

    for ind, group_cvr in enumerate(df_groupped_cvr):
        cvr_plan = group_cvr['sum_pl'].groupby(pd.Grouper(freq='Y')).agg('sum')
        group_cvr.drop(['sum_pl'], axis=1, inplace=True)

        cvr = pd.DataFrame(group_cvr['sum_ex'].resample('D').asfreq().fillna(0))

        cvr.columns = ['first_difference']
        cvr['cumsum'] = cvr['first_difference'].groupby(pd.Grouper(freq='Y')).cumsum()
        cvr['second_difference'] = cvr['first_difference'].groupby(pd.Grouper(freq='Y')).diff().fillna(0)

        cvr_id = group_cvr['cvr'][0]
        mdlParams['path_models'] = f'experiments/exp/cvr/exp_{num_exp}/{cvr_id}/models/'
        mdlParams['path_metrics'] = f'experiments/exp/cvr/exp_{num_exp}/{cvr_id}/csv/'
        mdlParams['path_plots'] = f'experiments/exp/cvr/exp_{num_exp}/{cvr_id}/plots/'

        if cvr.shape[0] < 365:
            skipped_codes.append({'cvr': cvr_id})
            save_model(f'{mdlParams["path_models"]}{skipped_codes_pkl}', skipped_codes)
            print(f'[SKIP] Skipped code: {cvr_id}')
            continue
        if state == 'load':
            pass
        else:
            mdlParams['id'] = cvr_id
            print(f'[INFO] Start index: {ind + 1} - id: {mdlParams["id"]}')

            _model = getattr(model, 'Model')
            model_cvr = _model(cvr, cvr_plan, weekends, mdlParams, model_name)

            start_date, end_date = pd.Timestamp('2021-01-10').date(), pd.Timestamp('2021-02-10').date()
            # predict over 10 month from 2021-01-10
            count = 10
            while count != 0:
                num_prediction = (end_date - start_date).days
                result = model_cvr.forecast(start_date, end_date, num_prediction)
                # save_sql_cvr(result, mdlParams['model_ver'])
                start_date = end_date
                end_date = (start_date + pd.DateOffset(months=1)).date()
                count -= 1

    return True


# noinspection DuplicatedCode
def predict_cvr_monthly(state: str, model_name: str, model_ver: str, data_from: str, force: bool, num_exp: int):
    mdlParams = {
        'path_data': 'data/',
        'path_skipped': f'experiments/exp/cvr/exp_{num_exp}/',
        'seed': 21,
        'plotForecast': True,
        'plotLoss': True,
        'plotPrediction': True,
        'plotTS': True,
        'silentPlot': True,
        'model_ver': model_ver,
        'period': 'monthly'
    }
    csv_name, cvr_codes_21, weekends_path = 'cvr.csv', 'cvr_codes_21.csv', f'{mdlParams["path_data"]}weekends.csv'
    skipped_codes_pkl = 'skipped_codes.pkl'

    df_cvr = read_sql_cvr() if data_from == 'sql' else pd.read_csv(f'{mdlParams["path_data"]}{csv_name}',
                                                                   index_col='docdate', parse_dates=['docdate'],
                                                                   infer_datetime_format=True)
    weekends = read_sql_weekend() if data_from == 'sql' else pd.read_csv(weekends_path, index_col='docdate',
                                                                         parse_dates=['docdate'],
                                                                         infer_datetime_format=True)
    cvr_codes = pd.read_csv(f"{mdlParams['path_data']}{cvr_codes_21}", header=None, dtype=int).iloc[:, 0].values

    # TODO aggregate over cvr
    aggregate = False
    if aggregate:
        df_groupped_cvr = df_cvr  # .loc[~df_cvr['cvr'].isin([110, 120])]
        group_cvr = df_groupped_cvr.loc[:, ['sum_ex', 'sum_pl']].groupby(df_groupped_cvr.index).agg('sum')

    input_cvr_codes = [610]
    if not input_cvr_codes:
        # If input codes is empty, add all codes
        input_cvr_codes = cvr_codes

    df_groupped_cvr = [group for _, group in df_cvr.groupby(['cvr']) if group['cvr'][0] in input_cvr_codes]

    del df_cvr
    os.makedirs(mdlParams["path_skipped"], exist_ok=True)
    skipped_codes = load_pkl(f'{mdlParams["path_skipped"]}{skipped_codes_pkl}')

    for ind, group_cvr in enumerate(df_groupped_cvr):
        cvr_plan = group_cvr['sum_pl'].groupby(pd.Grouper(freq='Y')).agg('sum')
        group_cvr.drop(['sum_pl'], axis=1, inplace=True)

        cvr = pd.DataFrame(group_cvr['sum_ex'].resample('D').asfreq().fillna(0))

        cvr.columns = ['first_difference']
        cvr['cumsum'] = cvr['first_difference'].groupby(pd.Grouper(freq='Y')).cumsum()
        cvr['second_difference'] = cvr['first_difference'].groupby(pd.Grouper(freq='Y')).diff().fillna(0)

        cvr_id = group_cvr['cvr'][0]
        mdlParams['path_models'] = f'experiments/exp/cvr/exp_{num_exp}/{cvr_id}/models/'
        mdlParams['path_metrics'] = f'experiments/exp/cvr/exp_{num_exp}/{cvr_id}/csv/'
        mdlParams['path_plots'] = f'experiments/exp/cvr/exp_{num_exp}/{cvr_id}/plots/'

        # Sum over 11 of current month to 10 of next month
        start_date, end_date = pd.Timestamp('2014-01-10').date(), pd.Timestamp('2014-02-10').date()
        cvr_true_month = pd.DataFrame()
        count = cvr[['first_difference']].groupby(pd.Grouper(freq='M')).sum().loc['2014-01-10':].shape[0]
        while count > 0:
            cvr_true_month_i = cvr[['first_difference']].loc[start_date + pd.Timedelta(days=1):end_date]
            cvr_true_month_sum = pd.Series(cvr_true_month_i.sum()['first_difference'], index=[start_date])
            cvr_true_month = pd.concat([cvr_true_month, cvr_true_month_sum], axis=0)
            start_date = end_date
            end_date = (start_date + pd.DateOffset(months=1)).date()
            count -= 1
        cvr_true_month.columns = ['first_difference']
        cvr_true_month.index = pd.Series(cvr_true_month.index).apply(lambda x: pd.Timestamp(x))

        if cvr_true_month.shape[0] < 48:
            skipped_codes.append({'cvr': cvr_id})
            save_model(f'{mdlParams["path_models"]}{skipped_codes_pkl}', skipped_codes)
            print(f'[SKIP] Skipped code: {cvr_id}')
            continue
        if state == 'load':
            pass
        else:
            mdlParams['id'] = cvr_id
            print(f'[INFO] Start index: {ind + 1} - id: {mdlParams["id"]}')

            _model = getattr(model, 'Model')
            model_cvr = _model(cvr_true_month, cvr_plan, weekends, mdlParams, model_name)

            start_date, end_date = pd.Timestamp('2021-01-10').date(), pd.Timestamp('2021-02-10').date()
            # predict over 10 month from 2021-01-10
            count = 10
            while count != 0:
                num_prediction = 1
                result = model_cvr.forecast(start_date, end_date, num_prediction)
                # save_sql_cvr(result, mdlParams['model_ver'])
                start_date = end_date
                end_date = (start_date + pd.DateOffset(months=1)).date()
                count -= 1

    return True


# noinspection DuplicatedCode
def predict_trend_with_seasonal(state: str, model_name: str, model_ver: str, data_from: str, force: bool, num_exp: int):
    mdlParams = {
        'path_data': 'data/',
        'path_skipped': f'experiments/exp/cvr/exp_{num_exp}/',
        'seed': 21,
        'plotForecast': True,
        'plotLoss': True,
        'plotPrediction': True,
        'plotTS': True,
        'silentPlot': True,
        'model_ver': model_ver,
        'period': 'daily'
    }
    csv_name, cvr_codes_21, weekends_path = 'cvr.csv', 'cvr_codes_21.csv', f'{mdlParams["path_data"]}weekends.csv'
    skipped_codes_pkl = 'skipped_codes.pkl'

    df_cvr = read_sql_cvr() if data_from == 'sql' else pd.read_csv(f'{mdlParams["path_data"]}{csv_name}',
                                                                   index_col='docdate', parse_dates=['docdate'],
                                                                   infer_datetime_format=True)
    cvr_codes = pd.read_csv(f"{mdlParams['path_data']}{cvr_codes_21}", header=None, dtype=int).iloc[:, 0].values

    # TODO aggregate over cvr
    aggregate = False
    if aggregate:
        df_groupped_cvr = df_cvr  # .loc[~df_cvr['cvr'].isin([110, 120])]
        group_cvr = df_groupped_cvr.loc[:, ['sum_ex', 'sum_pl']].groupby(df_groupped_cvr.index).agg('sum')

    input_cvr_codes = [610]
    if not input_cvr_codes:
        # If input codes is empty, add all codes
        input_cvr_codes = cvr_codes

    df_groupped_cvr = [group for _, group in df_cvr.groupby(['cvr']) if group['cvr'][0] in input_cvr_codes]

    del df_cvr
    os.makedirs(mdlParams["path_skipped"], exist_ok=True)
    skipped_codes = load_pkl(f'{mdlParams["path_skipped"]}{skipped_codes_pkl}')

    for ind, group_cvr in enumerate(df_groupped_cvr):
        cvr_plan = group_cvr['sum_pl'].groupby(pd.Grouper(freq='Y')).agg('sum')
        group_cvr.drop(['sum_pl'], axis=1, inplace=True)

        cvr = pd.DataFrame(group_cvr['sum_ex'].resample('D').asfreq().fillna(0))

        cvr.columns = ['first_difference']
        cvr['cumsum'] = cvr['first_difference'].groupby(pd.Grouper(freq='Y')).cumsum()
        cvr['second_difference'] = cvr['first_difference'].groupby(pd.Grouper(freq='Y')).diff().fillna(0)

        cvr_id = group_cvr['cvr'][0]
        mdlParams['path_models'] = f'experiments/exp/cvr/exp_{num_exp}/{cvr_id}/models/'
        mdlParams['path_metrics'] = f'experiments/exp/cvr/exp_{num_exp}/{cvr_id}/csv/'
        mdlParams['path_plots'] = f'experiments/exp/cvr/exp_{num_exp}/{cvr_id}/plots/'

        if cvr.shape[0] < 365:
            skipped_codes.append({'cvr': cvr_id})
            save_model(f'{mdlParams["path_models"]}{skipped_codes_pkl}', skipped_codes)
            print(f'[SKIP] Skipped code: {cvr_id}')
            continue
        if state == 'load':
            pass
        else:
            mdlParams['id'] = cvr_id
            print(f'[INFO] Start index: {ind + 1} - id: {mdlParams["id"]}')

            period = 12
            df_decomposed = seasonal_decompose(cvr[['first_difference']], model='additive', period=period)

            preds = []
            trend = df_decomposed.trend.dropna().copy()[:-4]
            trend.index = pd.Series(trend.index).apply(lambda x: pd.Timestamp(x))
            max_month_fc = 10
            month_count, ind_count = max_month_fc, 0
            while month_count != 0:
                last_forecast = preds[-1] if len(preds) > 0 else trend[-1]
                trend = trend.append(pd.Series(last_forecast,
                                               index=[cvr['first_difference'][-max_month_fc:].index.values[ind_count]]))
                model_i = linear_model.LinearRegression()
                model_i.fit(trend.index.map(pd.Timestamp.toordinal).values.reshape(-1, 1), trend.values)
                preds.append(model_i.predict(trend.index.map(pd.Timestamp.toordinal).values.reshape(-1, 1))[-1])
                month_count -= 1
                ind_count += 1

            fc = pd.Series(preds + df_decomposed.seasonal[-(max_month_fc + period):-max_month_fc].iloc[:len(preds)].values,
                           index=trend[-max_month_fc:].index)
            # pd.Series(preds, index=trend.loc['2021'].index).to_csv('trend.csv')
            pd.DataFrame(
                {
                    'fact': df_decomposed.observed,
                    'trend': df_decomposed.trend,
                    'seasonal': df_decomposed.seasonal,
                    'trend_fc': pd.Series(preds, index=trend[-max_month_fc:].index),
                    'season_fc': pd.Series(df_decomposed.seasonal[-(max_month_fc + period):-max_month_fc].iloc[:len(preds)].values,
                                           index=trend[-max_month_fc:].index),
                    'forecast': fc
                }).to_csv('seasonal.csv')

    return True
