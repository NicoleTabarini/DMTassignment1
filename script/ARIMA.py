import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})
from statsmodels.tsa.arima.model import ARIMA
from sktime.forecasting.model_selection import temporal_train_test_split
from math import sqrt
from sklearn.metrics import mean_squared_error





df = pd.read_csv("FINAL_FINAL.csv")
m = df["mean_mood"]
print(m)
plt.plot(m)
plot_acf(m)
#plt.show()

f = plt.figure()
ax1 = f.add_subplot (121)
ax1. set_title('1st Order Differencing')
ax1.plot(m.diff())

ax2 = f.add_subplot (122)
plot_acf(m.diff().dropna(), ax=ax2)
#plt. show()

f= plt. figure()
ax1 = f.add_subplot(121)
ax1.set_title("2nd Order Differencing")
ax1.plot(m.diff().diff())

ax2 = f.add_subplot(122)
plot_acf(m.diff().diff().dropna(),ax=ax2)
#plt. show()

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

#  looking at the partial autocorrelation plot, we see that the largest peak is always in position 1
#  => p=1

#  Looking at the number of lags crossing the threshold, we can determine how much of the past would be significant
#  enough to consider for the future. The ones with high correlation contribute more and would be enough to predict
#  future values
#  => q=2

# ARIMA(p=1, d=0, q= 2)
#print(m)
arima_model = ARIMA(m, order=(1,0,2))
model = arima_model.fit()
#print(model. summary())

#                                SARIMAX Results
# ==============================================================================
# Dep. Variable:                   mood   No. Observations:                 1268
# Model:                 ARIMA(1, 0, 2)   Log Likelihood               -1212.993
# Date:                Thu, 14 Apr 2022   AIC                           2435.985
# Time:                        19:36:02   BIC                           2461.711
# Sample:                             0   HQIC                          2445.650
#                                - 1268
# Covariance Type:                  opg
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const          6.9842      0.074     94.869      0.000       6.840       7.128
# ar.L1          0.9250      0.017     54.888      0.000       0.892       0.958
# ma.L1         -0.6072      0.028    -21.446      0.000      -0.663      -0.552
# ma.L2         -0.0989      0.027     -3.645      0.000      -0.152      -0.046
# sigma2         0.3965      0.011     37.462      0.000       0.376       0.417
# ===================================================================================
# Ljung-Box (L1) (Q):                   0.05   Jarque-Bera (JB):               495.17
# Prob(Q):                              0.82   Prob(JB):                         0.00
# Heteroskedasticity (H):               0.80   Skew:                            -0.59
# Prob(H) (two-sided):                  0.02   Kurtosis:                         5.82
# ===================================================================================

model.plot_diagnostics()
#plt. show()
#print(len(m))

patients = [y for x, y in df.groupby('id', as_index=False)]

for p in patients:
    training, test = temporal_train_test_split(p["mean_mood"],train_size=0.7,test_size=0.3)

    arima_model = ARIMA(training, order=(1,0,2))
    model = arima_model.fit()
    #print (model. summary())
    y_pred = model.forecast(len(test))
    y_true = test
    print("prediction",y_pred)
    print("real",test)
    residuals = pd.DataFrame(model.resid)
    # a line plot of the residual errors, suggesting that there may still be some trend information not captured by
    # the model
    residuals.plot()
    #plt.show()
    # density plot of residuals: density plot of the residual error values, suggesting the errors are Gaussian,
    # but may not be centered on zero
    residuals.plot(kind='kde')
    #plt.show()
    print(residuals.describe())
    # a non-zero mean in the residuals = there is a bias in the prediction
    rmse = sqrt(mean_squared_error(test, y_pred))
    print('Test RMSE: %.3f' % rmse)
    # plot forecasts against actual outcomes
    plt.plot(test)
    plt.plot(y_pred, color='red')
    #plt.show()
    mape = np.mean(np.abs(y_pred - test) / np.abs(test))  # Mean absolute percentage error
    mae = np.mean(np.abs(y_pred - test))
    # Mean absolute error
    mpe = np.mean((y_pred - test) / test)
    # Mean percentage error
    rmse = np.mean((y_pred - test) ** 2) ** .5
    # RMSE
    corr = np.corrcoef(y_pred, test)[0, 1]
    # Correlation Coefficient
    import pprint
    pprint.pprint({'mape':mape,
    "mae": mae,
    "mpe": mpe,
    "pmse": rmse,
    'corr' :corr})
