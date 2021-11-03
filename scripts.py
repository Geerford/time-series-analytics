import os

import pandas as pd
from sqlalchemy import create_engine

import model_catboost
from sql_scripts import read_sql_cvr, read_sql_weekend
from utils import load_pkl, save_model


def predict_cvr_month(state: str, model_name: str, model_ver: str, data_from: str, force: bool, num_exp: int):
    mdlParams = {
        'path_data': 'data/',
        'path_skipped': f'experiments/exp/cvr/exp_{num_exp}/',
        'seed': 21,
        'plotForecast': True,
        'plotLoss': True,
        'plotPrediction': True,
        'plotTS': True,
        'silentPlot': True,
        'model_ver': model_ver
    }
    csv_name, cvr_codes_21, weekends_path = 'cvr.csv', 'cvr_codes_21.csv', f'{mdlParams["path_data"]}weekends.csv'
    skipped_codes_pkl = 'skipped_codes.pkl'
    mdlParams['start_date'], mdlParams['end_date'] = pd.Timestamp('2021-01-10').date(), pd.Timestamp('2021-02-10').date()
    last_model_ind = 0
    df_cvr = read_sql_cvr() if data_from == 'sql' else pd.read_csv(f'{mdlParams["path_data"]}{csv_name}',
                                                                   index_col='docdate', parse_dates=['docdate'],
                                                                   infer_datetime_format=True)
    weekends = read_sql_weekend() if data_from == 'sql' else pd.read_csv(weekends_path, index_col='docdate',
                                                                         parse_dates=['docdate'],
                                                                         infer_datetime_format=True)
    cvr_codes = pd.read_csv(f"{mdlParams['path_data']}{cvr_codes_21}", header=None, dtype=int).iloc[:, 0].values
    df_groupped_cvr = [group for _, group in df_cvr.groupby(['cvr']) if group['cvr'][0] in cvr_codes]

    del df_cvr
    os.makedirs(mdlParams["path_skipped"], exist_ok=True)
    skipped_codes = load_pkl(f'{mdlParams["path_skipped"]}{skipped_codes_pkl}')

    for ind, group_cvr in enumerate(df_groupped_cvr[last_model_ind:]):
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
        if cvr_id != 610:
            continue
        if cvr.shape[0] < 365:
            skipped_codes.append({'cvr': cvr_id})
            save_model(f'{mdlParams["path_models"]}{skipped_codes_pkl}', skipped_codes)
            print(f'Skipped code: {cvr_id}')
            continue
        if state == 'load':
            pass
        else:
            mdlParams['id'] = cvr_id
            print(f'[INFO] Start index: {ind + 1} - id: {mdlParams["id"]}')

            model = getattr(model_catboost, 'Model') if model_name == 'catboost' else None
            model_cvr = model(cvr, cvr_plan, weekends, mdlParams)

            model_cvr.forecast_month()
    return True
