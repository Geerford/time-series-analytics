import pandas as pd

from decorator import Decorator


class Forecaster:
    def __init__(self, weekends: pd.DataFrame, num_prediction: int, decorator: Decorator = None, step: str = "1D"):
        """
        Constructor for the Forecaster object

        :param weekends: pd.DataFrame of weekends and work days
        :param num_prediction: number of time periods in the forecasting horizon
        :param decorator: Decorator object with the lags of the features
        :param step: str forecasting time period given as frequencies
        """

        self.weekends = weekends
        self.num_prediction = num_prediction
        self.step = step
        self.decorator = Decorator(weekends=self.weekends) if decorator is None else decorator

    def recursive(self, y: pd.DataFrame, model) -> pd.Series:
        """
        Multi-step recursive forecasting using the input time series data and a pre-trained model

        :param y: pd.DataFrame holding the input time-series to forecast
        :param model: already pre-trained regressor
        :return: pd.Series with forecasted values indexed by forecast horizon dates
        """

        dates = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=self.num_prediction, freq=self.step)
        forecasts = []
        target = y.copy()

        for date in dates:
            last_forecast = forecasts[-1] if len(forecasts) > 0 else 0.0
            target = target.append(pd.DataFrame(data=last_forecast, index=[date], columns=['t']))
            # Choose {self.num_prediction} time periods (~days) as forecasting horizon
            features, _ = self.decorator.generate_features(df=target, n_lags=self.num_prediction)

            predictions = model.predict(features)
            forecasts.append(predictions[-1])
        return pd.Series(data=forecasts, index=dates)

    def direct(self, y: pd.DataFrame, model_fn, params: dict = None) -> pd.Series:
        """
        Multi-step direct forecasting using a model to forecast each time period ahead

        :param y: pd.DataFrame holding the input time-series to forecast
        :param model_fn: a function for training the model which returns as output the trained model
                        cross-validation score and test score
        :param params: additional parameters for the training model
        :return: pd.Series with forecasted values indexed by forecast horizon dates
        """

        def one_step_features(date, step: str) -> (pd.DataFrame, pd.DataFrame):
            """
            One-step in direct forecasting

            :param date: Last date in target index
            :param step: str forecasting time period given as frequencies
            :return: pd.DataFrame with generated features
            :return pd.DataFrame with target
            """

            # Choose {self.num_prediction} time periods (~days) as forecasting horizon
            _features, _target = self.decorator.generate_features(df=y[y.index <= date], n_lags=self.num_prediction)
            # Build target to be ahead of the features built by the desired number of steps
            _target = y[y.index >= _features.index[0] + pd.Timedelta(days=step)]
            assert len(_features.index) == len(_target.index), '[ShapeError] Length(features) not equals to length(target)'

            return _features, _target

        params = params if params else {}
        dates = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=self.num_prediction, freq=self.step)
        forecasts = []

        forecast_features, _ = one_step_features(y.index[-1], 0)

        for n in range(1, self.num_prediction + 1):
            last_date = y.index[-1] - pd.Timedelta(days=n)
            features, target = one_step_features(last_date, n)

            model, cv_score, test_score = model_fn(features, target, **params)

            # Forecast {n}-steps ahead
            predictions = model.predict(forecast_features)
            forecasts.append(predictions[-1])

        return pd.Series(index=dates, data=forecasts)
