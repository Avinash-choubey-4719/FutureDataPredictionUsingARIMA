import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader.data as web
import datetime
import pandas as pd 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


btc = pd.read_csv("btc.csv")
print(btc.head())

btc.index = pd.to_datetime(btc['Date'], format='%Y-%m-%d')

del btc['Date']

sns.set()

plt.ylabel('BTC Price')
plt.xlabel('Date')
plt.xticks(rotation=45)

plt.plot(btc.index, btc['BTC-USD'], )

train = btc[btc.index < pd.to_datetime("2020-11-01", format='%Y-%m-%d')]
test = btc[btc.index > pd.to_datetime("2020-11-01", format='%Y-%m-%d')]

plt.plot(train, color = "black")
plt.plot(test, color = "red")
plt.ylabel('BTC Price')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.title("Train/Test split for BTC Data")
plt.show()

from statsmodels.tsa.statespace.sarimax import SARIMAX


y = train['BTC-USD']


ARMAmodel = SARIMAX(y, order = (1, 0, 1))


ARMAmodel = ARMAmodel.fit()


y_pred = ARMAmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = ARMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df["Predictions"] 


plt.plot(y_pred_out, color='green', label = 'Predictions')
plt.legend()


import numpy as np
from sklearn.metrics import mean_squared_error

arma_rmse = np.sqrt(mean_squared_error(test["BTC-USD"].values, y_pred_df["Predictions"]))
print("RMSE: ",arma_rmse)
