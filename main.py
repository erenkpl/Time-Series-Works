import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import numpy as np
import re
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from pmdarima.arima import auto_arima
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tools.eval_measures import rmse
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.ar_model import AutoReg
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("Covid 19.csv")

X = data.Date
Y = data.Confirmed

time_series = TimeSeriesSplit()
print(time_series)

data.Date = pd.to_datetime(data.Date)
data = data.set_index("Date")
print(data.head())

for train_index, test_index in time_series.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    train = data[:len(train_index)]
    test = data[len(train_index):len(train_index)+len(test_index)]

    # SARIMAX
    sarima_model = SARIMAX(train['Confirmed'], order=(4, 0, 2), seasonal_order=(2, 1, [], 11))
    sarima_result = sarima_model.fit()
    sarime_pred = sarima_result.predict(start=len(train), end=len(data) - 1, typ="levels").rename("SARIMA Tahmini")
    data['Confirmed'].plot(figsize=(15, 6), legend=True)
    test['Confirmed'].plot(figsize=(15, 5), legend=True)
    sarime_pred.plot(legend=True)

    # AR
    ar_model = AutoReg(train['Confirmed'], lags=58).fit()
    pred = ar_model.predict(start=len(train), end=len(data) - 1, dynamic=False).rename("AR tahmini")
    test['Confirmed'].plot(figsize=(15, 5), legend=True)
    pred.plot(legend=True)

    # ARIMA
    model_arima = auto_arima(train, seasonal=True, D=None, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    model_arima = model_arima.fit(train)
    prediction, confint = model_arima.predict(n_periods=len(test), return_conf_int=True)
    cf = pd.DataFrame(confint)
    prediction_series = pd.Series(prediction, index=test.index)
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(data.Confirmed)
    ax.plot(prediction_series)
    ax.fill_between(prediction_series.index, cf[0], cf[1], color='grey', alpha=.3)
    model_arima.plot_diagnostics(figsize=(14, 10))

    model_arima.summary()
    get_param = model_arima.get_params()
    order_a = get_param.get('order')
    order_s = get_param.get('seasonal_order')

    arima_model = sm.tsa.ARIMA(train['Confirmed'], order=order_a, seasonal_order=order_s)
    arima_result = arima_model.fit()
    arima_pred = arima_result.predict(start=len(train), end=len(data) - 1, typ="levels").rename("ARIMA Tahmini")
    arima_pred.plot(legend=True)

    # ARMA
    arma_model = SARIMAX(train['Confirmed'], order=(4, 0, 2), seasonal_order=(2, 1, [], 11))
    arma_result = arma_model.fit()
    arma_pred = arma_result.predict(start=len(train), end=len(data) - 1, typ="levels").rename("ARMA Tahmini")
    arma_pred.plot(legend=True)

    # ETS
    model = ETSModel(train['Confirmed'])
    fit = model.fit()
    model_pred = fit.predict(start=len(train), end=len(data) - 1).rename("ETS Tahmini")
    model_pred.plot(legend=True)

    plt.show()

vis = go.Scatter(x=data.index, y=data.Confirmed)
py.plot([vis])

rmse_sarima = rmse(test['Confirmed'], sarime_pred)
print('RMSE: %.3f' % rmse_sarima)

sarima_result.summary()

plt.show()

data['Confirmed'].plot(figsize=(15, 6), legend=True)

yeni_sarima_tahmin = sarima_result.predict(start=840, end=1000, dynamic=True).rename("SARIMA Tahmini")
yeni_sarima_tahmin.plot(legend=True)

yeni_ar_tahmini = ar_model.predict(start=701, end=1000, dynamic=True).rename('AR Tahmini')
yeni_ar_tahmini.plot(legend=True)

yeni_arima_tahmini = arima_result.predict(start=840, end=1000, dynamic=True).rename("ARIMA Tahmini")
yeni_arima_tahmini.plot(legend=True)

yeni_arma_tahmini = arma_result.predict(start=840, end=1000, dynamic=True).rename("ARMA Tahmini")
yeni_arma_tahmini.plot(legend=True)

yeni_ets_tahmini = fit.predict(start=840, end=1000).rename("ETS Tahmini")
yeni_ets_tahmini.plot(legend=True)

plt.show()

def mape(actual, prediction):
    return np.mean(np.abs((actual-prediction) / actual)) * 100

result = mape(test['Confirmed'], sarime_pred)
print("MAPE (SARIMAX): ", result)
result_ar = mape(test['Confirmed'], pred)
print("MAPE (AR): ", result_ar)
result_arima = mape(test['Confirmed'], prediction)
print("MAPE (ARIMA): ", result_arima)
result_arma = mape(test['Confirmed'], arma_pred)
print("MAPE (ARMA): ", result_arma)
result_ets = mape(test['Confirmed'], model_pred)
print("MAPE (ETS): ", result_ets)
