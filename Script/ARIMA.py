import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})
from statsmodels.tsa.arima.model import ARIMA
from sktime.forecasting.model_selection import temporal_train_test_split
from math import sqrt
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
from pmdarima.arima import auto_arima





df = pd.read_csv("FINAL_FINAL.csv")
df_date=df[["date","mean_mood"]]
new_df=df_date.groupby(by="date",as_index=False).median()
m = new_df["mean_mood"]


print(m)
plt.plot(m)
plot_acf(m)
plot_pacf(m)


# AutoARIMA
print(auto_arima(m))
# -> 0,0,1


# Search for d
f = plt.figure()
ax1 = f.add_subplot (121)
ax1. set_title('1st Order Differencing')
ax1.plot(m.diff())

ax2 = f.add_subplot (122)
plot_acf(m.diff().dropna(), ax=ax2)


f= plt. figure()
ax1 = f.add_subplot(121)
ax1.set_title("2nd Order Differencing")
ax1.plot(m.diff().diff())

ax2 = f.add_subplot(122)
plot_acf(m.diff().diff().dropna(),ax=ax2)


result = adfuller(m.dropna())
print('p-value:',result[1])
result = adfuller (m.diff(). dropna())
print('p-value:', result[1])
result= adfuller(m.diff().diff().dropna())
print('p-value:', result[1])
#  p-value: 9.08803422901021e-08
#  p-value: 3.676410155970776e-22
#  p-value: 1.440512308425612e-26
#  => d=0



# Model creation
training, test = temporal_train_test_split(m, train_size=0.7, test_size=0.3)
training.reset_index(inplace=True, drop=True)
test.reset_index(inplace=True, drop=True)
arima_model = ARIMA(training, order=(0,0,1))
t=len(test)
for i in range(t):
    arima_model = ARIMA(training, order=(0,0,1))
    model = arima_model.fit()
    model.plot_diagnostics()
    y_pred = model.forecast(1)
    try:pred = pred.append(pd.Series(y_pred))
    except: pred=y_pred

    training=training.append(pd.Series(test.iloc[i]))



# Plot performance prediction vs actual
pred.reset_index(inplace=True, drop=True)
print(test)
plt.figure(figsize=(12,5), dpi=100)
plt.plot(test, label='actual')
plt.plot(pred, label='ARIMA prediction')
plt.legend()
plt.show()


# Evaluate performance
print("mae",mean_absolute_error(test,pred))
print("mape",mean_absolute_percentage_error(test,pred))
rmse = sqrt(mean_squared_error(test, pred))
print("rmse",rmse)
print("wmape",sum(np.abs(test - pred))/sum(test))
