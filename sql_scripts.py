import datetime
import re

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from utils import load_yaml


def read_sql_cvr():
    """
    Read SQL query into a DataFrame.

    :return: pd.DataFrame from database
    """
    params = load_yaml("db_credentials")
    engine = create_engine(f"{params['dialect']}://{params['username']}:"
                           f"{params['password']}@{params['host']}:{params['port']}/{params['database']}", echo=True)
    df_exp = pd.concat([
        # pd.read_sql('SELECT sum_ex, sum_pl, cvr, fb_flag, docdate FROM exp_data_11', engine, index_col='docdate'),
        # pd.read_sql('SELECT sum_ex, sum_pl, cvr, fb_flag, docdate FROM exp_data_12', engine, index_col='docdate'),
        # pd.read_sql('SELECT sum_ex, sum_pl, cvr, fb_flag, docdate FROM exp_data_13', engine, index_col='docdate'),
        pd.read_sql('SELECT sum_ex, sum_pl, cvr, fb_flag, docdate FROM exp_data_14', engine, index_col='docdate'),
        pd.read_sql('SELECT sum_ex, sum_pl, cvr, fb_flag, docdate FROM exp_data_15', engine, index_col='docdate'),
        pd.read_sql('SELECT sum_ex, sum_pl, cvr, fb_flag, docdate FROM exp_data_16', engine, index_col='docdate'),
        pd.read_sql('SELECT sum_ex, sum_pl, cvr, fb_flag, docdate FROM exp_data_17', engine, index_col='docdate'),
        pd.read_sql('SELECT sum_ex, sum_pl, cvr, fb_flag, docdate FROM exp_data_18', engine, index_col='docdate'),
        pd.read_sql('SELECT sum_ex, sum_pl, cvr, fb_flag, docdate FROM exp_data_19', engine, index_col='docdate'),
        pd.read_sql('SELECT sum_ex, sum_pl, cvr, fb_flag, docdate FROM exp_data_20', engine, index_col='docdate'),
        pd.read_sql('SELECT sum_ex, sum_pl, cvr, fb_flag, docdate FROM exp_data_21', engine, index_col='docdate')
    ])

    df_exp['cvr'] = [re.sub('^\\s+|\\s+$', '', x, flags=re.UNICODE) for x in df_exp['cvr']]
    df_exp['cvr'] = [int(f'{str(x[:2])}0') for x in df_exp['cvr']]
    df_exp = df_exp.groupby([df_exp.index, 'fb_flag', 'cvr']).agg({'sum_ex': 'sum',
                                                                   'sum_pl': 'sum'}).reset_index().set_index('docdate')
    df_fb_0 = df_exp.loc[df_exp.fb_flag == '0'].drop('fb_flag', axis=1)
    df_fb_0.index = pd.to_datetime(df_fb_0.index)
    return df_fb_0


def read_sql_weekend():
    params = load_yaml("db_credentials")
    engine = create_engine(f"{params['dialect']}://{params['username']}:"
                           f"{params['password']}@{params['host']}:{params['port']}/{params['database']}", echo=True)
    df = pd.concat([
        # pd.read_sql('SELECT flag, dbn FROM dbn_table_10', engine, index_col='dbn'),
        # pd.read_sql('SELECT flag, dbn FROM dbn_table_11', engine, index_col='dbn'),
        # pd.read_sql('SELECT flag, dbn FROM dbn_table_12', engine, index_col='dbn'),
        # pd.read_sql('SELECT flag, dbn FROM dbn_table_13', engine, index_col='dbn'),
        pd.read_sql('SELECT flag, dbn FROM dbn_table_14', engine, index_col='dbn'),
        pd.read_sql('SELECT flag, dbn FROM dbn_table_15', engine, index_col='dbn'),
        pd.read_sql('SELECT flag, dbn FROM dbn_table_16', engine, index_col='dbn'),
        pd.read_sql('SELECT flag, dbn FROM dbn_table_17', engine, index_col='dbn'),
        pd.read_sql('SELECT flag, dbn FROM dbn_table_18', engine, index_col='dbn'),
        pd.read_sql('SELECT flag, dbn FROM dbn_table_19', engine, index_col='dbn'),
        pd.read_sql('SELECT flag, dbn FROM dbn_table_20', engine, index_col='dbn'),
        pd.read_sql('SELECT flag, dbn FROM dbn_table_21', engine, index_col='dbn')])
    df.index = pd.to_datetime(df.index)
    return pd.DataFrame(np.logical_xor(pd.to_numeric(df['flag'], downcast='unsigned'), 1).astype(int)).rename(columns={0: 'flag'})


def save_sql_cvr(forecast: pd.DataFrame, model_ver: str):
    forecast['model'] = model_ver
    params = load_yaml("db_credentials")
    engine = create_engine(f"{params['dialect']}://{params['username']}:"
                           f"{params['password']}@{params['host']}:{params['port']}/{params['database']}", echo=True)
    forecast.to_sql('cvr_test_forecast', engine, if_exists='append', index=False)
    model_list = pd.DataFrame({
        'model_name': model_ver,
        'model_use': params['experiment_desc'],
        'owner': params['owner'],
        'source': datetime.date.today()
    })
    model_list.to_sql('model_list', engine, if_exists='append', index=False)
