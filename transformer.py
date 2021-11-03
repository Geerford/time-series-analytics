import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class Transformer:
    def __init__(self, log=False, detrend=False, diff=False, scale=False):
        self.log = log
        self.log_max = None
        self.detrend = detrend
        self.trend = pd.Series(dtype=np.float64) if self.detrend else None
        self.diff = diff
        self.scale = scale
        self.scaler = StandardScaler() if self.scale else None

    def transform(self, y: pd.DataFrame) -> pd.DataFrame:
        if self.diff:
            y = y.groupby(pd.Grouper(freq='Y')).diff().fillna(0)
        if self.detrend:
            self.trend = self.get_trend(y) - np.mean(y.values)
            y = y.subtract(self.trend)
        if self.log:
            self.log_max = max(np.abs(y[y.columns[0]])) + 1
            y += self.log_max
            y = pd.DataFrame(np.log(y), index=y.index, columns=y.columns)
        if self.scale:
            y = pd.DataFrame(self.scaler.fit_transform(y), index=y.index, columns=y.columns)
        return y

    def inverse(self, y: pd.DataFrame) -> pd.DataFrame:
        if self.scale:
            y = pd.DataFrame(self.scaler.inverse_transform(y), index=y.index, columns=y.columns)
        if self.log:
            y = pd.DataFrame(np.exp(y), index=y.index, columns=y.columns)
            y -= self.log_max
        if self.detrend:
            try:
                assert len(y.index) == len(self.trend.index)
                y += self.trend
            except AssertionError:
                print("Use a different transformer for each target to transform")
        if self.diff:
            y = y.groupby(pd.Grouper(freq='Y')).cumsum()
        return y

    @staticmethod
    def get_trend(df: pd.DataFrame) -> pd.DataFrame:
        """
        Get the linear trend on the data which makes the time series not stationary
        """
        n = len(df.index)
        X = np.reshape(np.arange(0, n), (n, 1))
        y = np.array(df)
        model = LinearRegression()
        model.fit(X, y)
        trend = model.predict(X)
        return pd.DataFrame(index=df.index, data=trend, columns=df.columns)
