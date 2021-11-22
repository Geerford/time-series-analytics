import os
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import yaml


def max_absolute_error(y_true: np.array, y_pred: np.array) -> float:
    """
    Max of mean absolute error regression loss

    :param y_true: Ground truth (fact) target values.
    :param y_pred: Estimated target values.
    :return: float
    """
    assert len(y_true) == len(y_pred), '[ShapeError] Length(y_true) not equals to length(y_pred)'

    return max([np.abs(true-pred) for true, pred in zip(y_true, y_pred)])


def symmetric_mean_absolute_percentage_error(y_true: np.array, y_pred: np.array) -> float:
    """
    Symmetric mean absolute percentage error regression loss

    :param y_true: Ground truth (fact) target values.
    :param y_pred: Estimated target values.
    :return: float in the range [0, 100]
    """
    assert len(y_true) == len(y_pred), '[ShapeError] Length(y_true) not equals to length(y_pred)'

    div = np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))
    div[np.isnan(div)] = 0
    return np.sum(div) / len(y_true)


def time_series_thresholds(series: pd.Series) -> list:
    """
    Get thresholds of anomalies

    :param series: pd.Series holding the input time-series target
    :return: list of thresholds
    """
    return [
        3 * np.std(series, ddof=1),
        4 * np.std(series, ddof=1),
        5 * np.std(series, ddof=1),
        np.percentile(series, 90),
        np.percentile(series, 95),
        None
    ]


def plot_go(traces: list, title: str, silent: bool, errors: list = None, save_path: str = None):
    """
    Show and save a figure of list<go.Scatter> using default renderer

    :param traces: list of time series like trace[x, y, name]
    :param title: The title of figure
    :param silent: if True show the plots
    :param errors: list of metrics like metrics[MAE, MAPE, SMAPE]
    :param save_path: str of path for saving plots
    """
    scatters = []
    for trace in traces:
        scatters.append(go.Scatter(
            x=trace[0],
            y=trace[1],
            mode='lines',
            name=trace[2]
        ))
    if errors:
        layout = go.Layout(
            title=f'{title} <br>MAE: {errors[0]: >15} MAPE: {errors[1]: >5} SMAPE: {errors[2]: >5}',
            xaxis={'title': 'Date'},
            yaxis={'title': 'Difference'}
        )
    else:
        layout = go.Layout(
            title=title,
            xaxis={'title': 'Date'},
            yaxis={'title': 'Difference'}
        )
    fig = go.Figure(data=scatters, layout=layout)
    if save_path:
        fig.write_image(save_path)
    if not silent:
        fig.show()


def load_pkl(model_name: str):
    """
    Load pickle file

    :param model_name: name of file
    """
    if os.path.isfile(model_name):
        with open(model_name, 'rb') as f:
            params = pickle.load(f)
            return params


def save_model(model_name: str, model):
    """
    Save model as pickle file

    :param model_name: name of file
    :param model: object to save
    """
    if os.path.isfile(model_name):
        os.remove(model_name)
    with open(model_name, 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


def load_yaml(config_name: str):
    """
    Load pickle file

    :param config_name: name of config file
    """
    with open(f'config/{config_name}.yaml', 'r') as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)
    return params


def save_yaml(params: dict, config_name: str):
    """
    Save model as pickle file

    :param params: dict to save
    :param config_name: name of config file
    """
    with open(f'config/{config_name}.yaml', 'w') as stream:
        try:
            yaml.safe_dump(params, stream, default_flow_style=False)
        except yaml.YAMLError as e:
            print(e)
