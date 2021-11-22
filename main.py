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
                        default="predict_cvr",
                        choices=["predict_cvr", "predict_cvr_threshold"],
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

    # parser.add_argument("-f", "--force",
    #                     type=bool,
    #                     default=True,
    #                     choices=[True, False],
    #                     help="The force saving to database")

    parser.add_argument("-hf", "--horizon",
                        type=str,
                        default="daily",
                        choices=["daily", "monthly"],
                        help="The horizon of forecasting")

    parser.add_argument("-sd", "--seasonal_decompose",
                        type=bool,
                        default=False,
                        choices=[True, False],
                        help="The seasonal decomposition of time series")

    parser.add_argument("-e", "--experiment",
                        type=int,
                        default=0,
                        help="The number of experiment")

    parser.add_argument("-rs", "--random_seed",
                        type=int,
                        default=21,
                        help="The number of random seed")

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
    configure_run(seed=args.random_seed)

    result = getattr(scripts, args.script_name)(args.state, args.model, args.version, args.data, args.experiment,
                                                args.horizon, args.seasonal_decompose)
    sys.exit()
