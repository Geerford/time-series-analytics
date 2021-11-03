import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import periodogram


class Describer:
    @staticmethod
    def describe(df: pd.DataFrame):
        """
        Perform all the methods that described below

        :param df: pd.DataFrame holding the time series values
        """

        Describer.hist(df)
        Describer.scatter(df)
        Describer.qqplot(df)
        Describer.decompose(df)
        Describer.dickey_fuller_test(df)
        Describer.resample_plot(df)
        Describer.periodogram(df)

    @staticmethod
    def scatter(df: pd.DataFrame):
        """
        Draw a scatter plot of each DataFrame's columns

        :param df: pd.DataFrame holding the time series values
        """

        for column in df.columns:
            sns.scatterplot(x=df.index, y=df[column])
            plt.show()

        print(f"\n\033[4mStats {'': <50}\033[0m")
        print(df.describe())

    @staticmethod
    def hist(df: pd.DataFrame):
        """
        Draw histogram of each DataFrame's columns and aggregate skewness values

        :param df: pd.DataFrame holding the time series values
        """

        for column in df.columns:
            sns.histplot(df[column], kde=True, bins=50)
            plt.show()

        print(f"\n\033[4mSkewness {'': <50}\033[0m")
        print(df.agg(['skew']).transpose()['skew'])

    @staticmethod
    def qqplot(df: pd.DataFrame):
        """
        Draw Q-Q plot of the quantiles of each DataFrame's columns versus the quantiles of a distribution.

        :param df: pd.DataFrame holding the time series values
        """

        for column in df.columns:
            sm.qqplot(df[column], line='q')
            plt.show()

        print(f"\n\033[4mSkewness {'': <50}\033[0m")
        print(df.agg(['skew']).transpose()['skew'])

    @staticmethod
    def decompose(df: pd.DataFrame):
        """
        Draw seasonal decomposition using moving average of each DataFrame's columns

        :param df: pd.DataFrame holding the time series values
        """

        for column in df.columns:
            seasonal_decompose(df[column], model='additive').plot()
            plt.show()

    @staticmethod
    def resample_plot(df: pd.DataFrame):
        for column in df.columns:
            df[column].resample('W').agg(['sum', 'mean', 'std']).plot(subplots=True, legend=True,
                                                                      title=f'{column} resampled over week')
            df[column].resample('Q').agg(['sum', 'mean', 'std']).plot(subplots=True, legend=True,
                                                                      title=f'{column} resampled over quarter')
            plt.show()

    @staticmethod
    def dickey_fuller_test(df: pd.DataFrame, lags: int = 90):
        """
        Draw augmented Dickey-Fuller test with acf and pacf of each DataFrame's columns

        :param df: pd.DataFrame holding the time series values
        :param lags: lags of the last period df
        """

        for column in df.columns:
            layout = (2, 2)
            ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
            acf_ax = plt.subplot2grid(layout, (1, 0))
            pacf_ax = plt.subplot2grid(layout, (1, 1))

            ts_ax.plot(df[column])
            adfuller = sm.tsa.stattools.adfuller(df[column])

            print(f"\n\033[4m{column}. ADF Statistic {'': <27}\033[0m")
            print(adfuller[0])

            print(f"\n\033[4m{column}. Critical Values {'': <25}\033[0m")
            for key, value in adfuller[4].items():
                print(f'\t{key}: {value:.3f}')

            p_value = adfuller[1]
            ts_ax.set_title(f'{column}\nDickey-Fuller: p={p_value:.8f}')
            smt.graphics.plot_acf(df[column], lags=lags, ax=acf_ax)
            smt.graphics.plot_pacf(df[column], method='ywm', lags=lags, ax=pacf_ax)
            plt.tight_layout()
            plt.show()

    @staticmethod
    def periodogram(df: pd.DataFrame):
        """
        Draw power spectral density using a periodogram of each DataFrame's columns

        :param df: pd.DataFrame holding the time series values
        """

        for column in df.columns:
            # fs - sampling frequency of df
            freqencies, spectrum = periodogram(df[column], fs=pd.Timedelta(days=365) / pd.Timedelta(days=1),
                                               detrend='linear', window='boxcar', scaling='spectrum')
            _, ax = plt.subplots()
            ax.step(freqencies, spectrum)
            ax.set_xscale("log")
            periods = [1, 2, 4, 6, 12, 26, 52, 104]
            ax.set_xticks(periods)
            ax.set_xticklabels([
                f"Annual ({periods[0]})",
                f"Semiannual ({periods[1]})",
                f"Quarterly ({periods[2]})",
                f"Bimonthly ({periods[3]})",
                f"Monthly ({periods[4]})",
                f"Biweekly ({periods[5]})",
                f"Weekly ({periods[6]})",
                f"Semiweekly ({periods[7]})",
            ], rotation=30)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.set_ylabel("Variance")
            ax.set_title(f"Periodogram {column}")
            plt.show()
