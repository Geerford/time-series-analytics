import os
import pickle
import numpy as np
import plotly.graph_objs as go


def max_absolute_error(y_true: np.array, y_pred: np.array) -> float:
    assert len(y_true) == len(y_pred), '[ShapeError] Length(y_true) not equals to length(y_pred)'

    return max([np.abs(true-pred) for true, pred in zip(y_true, y_pred)])


def symmetric_mean_absolute_percentage_error(y_true: np.array, y_pred: np.array) -> float:
    assert len(y_true) == len(y_pred), '[ShapeError] Length(y_true) not equals to length(y_pred)'

    div = np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))
    div[np.isnan(div)] = 0
    return np.sum(div) / len(y_true)


def plot_go(traces: list, title: str, silent: bool, errors: list = None, save_path: str = None):
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


def load_pkl(model_name):
    if os.path.isfile(model_name):
        with open(model_name, 'rb') as f:
            params = pickle.load(f)
            return params


def save_model(model_name, model):
    if os.path.isfile(model_name):
        os.remove(model_name)
    with open(model_name, 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
