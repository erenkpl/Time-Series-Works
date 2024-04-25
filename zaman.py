import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

data = pd.read_csv('Covid 19.csv')

dateparse = lambda dates : pd.datetime.strptime(dates, '%Y-%m-%d')
data = pd.read_csv('Covid 19.csv', parse_dates=['Date'], index_col='Date')

ts = data['Confirmed']

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/austa.csv')
fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.values.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0, 1.2))
plot_acf(df.value.diff().dropna(), ax=axes[1])


def test_stationary(timeseries):
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()

    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

test_stationary(ts)
plt.show()
