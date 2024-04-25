import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pandas import Series
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings

warnings.filterwarnings("ignore")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train_original = train.copy()
test_original = test.copy()

train.columns, test.columns
train.dtypes, test.dtypes
train.shape, test.shape

train['Datetime'] = pd.to_datetime(train.date, format='%Y-%m-%d')
test['Datetime'] = pd.to_datetime(test.date, format='%Y-%m-%d')
train_original['Datetime'] = pd.to_datetime(train_original.date, format='%Y-%m-%d')
test_original['Datetime'] = pd.to_datetime(test_original.date, format='%Y-%m-%d')

for i in (train, test, train_original, test_original):
    i['year'] = i.Datetime.dt.year
    i['month'] = i.Datetime.dt.month
    i['day'] = i.Datetime.dt.day

def applyer (row):
    if row.dayofweek == 5 or row.dayofweek == 6:
        return 1
    else:
        return 0

temp2 = train['Datetime'].apply(applyer)
train['weekend']=temp2

train.index = train['Datetime']
df = train.drop('id', 1)
ts = df['sales']
plt.figure(figsize=(16, 8))
plt.plot(ts, label='Count')
plt.title('Time Series')
plt.xlabel("Time(year-month)")
plt.ylabel("Count")
plt.legend(loc='best')

train.Timestamp = pd.to_datetime(train.date, format='%Y-%m-%d')
train.index = train.Timestamp
daily = train.resample("D").mean()
weekly = train.resample("W").mean()
monthly = train.resample("M").mean()
yearly = train.resample("Y").mean()

fig, axs = plt.subplots(4,1)
daily.sales.plot(figsize=(15, 8), title='Daily', fontsize=14, ax=axs[0])
weekly.sales.plot(figsize=(15, 8), title='Weekly', fontsize=14, ax=axs[1])
monthly.sales.plot(figsize=(15, 8), title='Monthly', fontsize=14, ax=axs[2])
yearly.sales.plot(figsize=(15, 8), title='Yearly', fontsize=14, ax=axs[3])

test.Timestamp = pd.to_datetime(test.date, format='%Y-%m-%d')
test.index = test.Timestamp
test = test.resample('D').mean()
train.Timestamp = pd.to_datetime(train.date, format='%Y-%m-%d')
train.index = train.Timestamp
train = train.resample('D').mean()

Train = train['2013-01-01':'2017-05-15']
valid = train['2017-05-16':'2017-08-15']
Train.sales.plot(figsize=(15, 8), title='Daily sales', fontsize=14, label='train')
valid.sales.plot(figsize=(15, 8), title='Daily sales', fontsize=14, label='valid')

dd = np.asarray(Train.sales)
y_hat = valid.copy()
y_hat['naive'] = dd[len(dd)-1]
plt.figure(figsize=(12, 8))
plt.plot(Train.index, Train['sales'], label='Train')
plt.plot(valid.index, valid['sales'], label='Valid')
plt.plot(y_hat.index, y_hat['naive'], label='Native Forecast')
plt.legend(loc='best')
plt.title("Native Forecast")
plt.show()

rms = sqrt(mean_squared_error(valid.sales, y_hat.naive))
print(rms)
