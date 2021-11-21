import argparse
import os
import random as rn
import sys

import numpy as np

import scripts


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--state",
                        type=str,
                        default='train',
                        choices=['train', 'load'],
                        help="The mode of model")

    parser.add_argument("-sn", "--script_name",
                        type=str,
                        default="predict_cvr_daily",
                        choices=["predict_cvr_daily", "predict_cvr_monthly", "predict_cvr_daily_anomaly",
                                 "predict_trend_with_seasonal"],
                        help="The name of script")

    parser.add_argument("-m", "--model",
                        type=str,
                        default="catboost",
                        choices=["catboost", "sarimax"],
                        help="The type of model")

    parser.add_argument("-v", "--version",
                        type=str,
                        help="The version of model")

    parser.add_argument("-d", "--data",
                        type=str,
                        default="csv",
                        choices=["csv", "sql"],
                        help="The type of data load")

    parser.add_argument("-f", "--force",
                        type=bool,
                        default=True,
                        choices=[True, False],
                        help="The force saving to database")

    parser.add_argument("-e", "--experiment",
                        type=int,
                        default=0,
                        help="The number of experiment")

    params = parser.parse_args()
    assert params.script_name in dir(scripts), 'Unknown function name, check in scripts module'

    return params


def configure_run(seed: int):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    np.seterr(all="ignore")
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)
    rn.seed(seed)


if __name__ in '__main__':
    args = read_args()
    configure_run(seed=21)

    result = getattr(scripts, args.script_name)(args.state, args.model, args.version, args.data, args.force,
                                                args.experiment)
    sys.exit()
