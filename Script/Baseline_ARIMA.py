import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sktime.forecasting.model_selection import temporal_train_test_split
from math import sqrt
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
from sktime.forecasting.naive import NaiveForecaster



df = pd.read_csv("FINAL_FINAL.csv")
df_date=df[["date","mean_mood"]]
new_df=df_date.groupby(by="date",as_index=False).median()
m = new_df["mean_mood"]



# Build the model
training, test = temporal_train_test_split(m, train_size=0.7, test_size=0.3)
forecaster = NaiveForecaster(strategy="last")
t=len(test)
for i in range(t):
    forecaster.fit(training)
    y_pred = forecaster.predict(1)
    try:pred = pred.append(pd.Series(y_pred))
    except: pred=y_pred
    training=training.append(pd.Series(test.iloc[i]))
    training.reset_index(inplace=True, drop=True)



# Evaluate  the  model
print("mae",mean_absolute_error(test,pred))
print("mape",mean_absolute_percentage_error(test,pred))
rmse = sqrt(mean_squared_error(test, pred))
print("rmse",rmse)
print("wmape",sum(np.abs(test - pred))/sum(test))




# Plot model performance vs actual values
test.reset_index(inplace=True, drop=True)
pred.reset_index(inplace=True, drop=True)
plt.figure(figsize=(12,5), dpi=100)
plt.plot(test, label='actual')
plt.plot(pred, label='baseline prediction')
plt.legend()
plt.show()
