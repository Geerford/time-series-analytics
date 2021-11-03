import argparse
import os
import random as rn
import sys

import numpy as np
from tensorboard import program

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
                        default="predict_cvr_month",
                        choices=["predict_cvr_month"],
                        help="The name of script")

    parser.add_argument("-m", "--model",
                        type=str,
                        default="catboost",
                        choices=["catboost"],
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


def configure_run(seed: int) -> str:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    np.seterr(all="ignore")
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)
    rn.seed(seed)
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', 'catboost_info'])
    url = tb.launch()
    return url


if __name__ in '__main__':
    args = read_args()
    tb_url = configure_run(seed=21)
    print(f'[INFO] Tensorboard starting up {tb_url}')

    result = getattr(scripts, args.script_name)(args.state, args.model, args.version, args.data, args.force,
                                                args.experiment)
    sys.exit()
