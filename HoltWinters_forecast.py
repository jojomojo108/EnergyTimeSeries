import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.dates as mdates

# first in-data forecast (to check model behaviour on known in-data points)
# then actual out-data forecast into unknown future

# which data column is prediction target
predict_col = 'ENERGY_PRODUCED'

# load relevant data
df = pd.read_csv('PV_system.csv', usecols=['Time', predict_col], index_col='Time', parse_dates=True)

# time data every 15 minutes
freq = '15min'
df.index.freq = freq

# how many 15 minutes per day
quarters_per_day = int(24 * 60 / 15)
days_ahead = 2
# forecast how many steps ahead
Ntest = int(days_ahead*quarters_per_day)

# split data set into train and test sub data sets
train = df.iloc[:-Ntest]
test = df.iloc[-Ntest:]

# indices to later store predictions into dataframe
train_idx = df.index <= train.index[-1]
test_idx = df.index > train.index[-1]

# instantiate Holt-Winters model for in-data forecast
holtwinters_indata = ExponentialSmoothing(
                     train[predict_col],
                     initialization_method='heuristic',
                     trend='add', # 'add' for linear, 'mul' for curve
                     seasonal='add', # 'add' for consant cycles, 'mul' for changing cycles
                     seasonal_periods=quarters_per_day) # cycle period is 1 day = 96 quarters

# fit model to training data
res_holtwinters_indata = holtwinters_indata.fit()

# save fitted data
df.loc[train_idx, 'Holt-Winters-in-data'] = res_holtwinters_indata.fittedvalues

# do and save in-data forecast
df.loc[test_idx, 'Holt-Winters-in-data'] = res_holtwinters_indata.forecast(Ntest)
print(res_holtwinters_indata.mle_retvals)

# plot fitted data and in-data forecast
fig, ax = plt.subplots(figsize=(11,6))
ax.plot(df[predict_col], label='data', color='purple')
ax.plot(df.loc[train_idx, 'Holt-Winters-in-data'], label='in-data fit', linestyle='dotted', color='orange')
ax.plot(df.loc[test_idx, 'Holt-Winters-in-data'], label='in-data forecast', linestyle='dashed', color='orange')
ax.set_ylabel('Energy produced (Wh)')
ax.set_xlabel('Date')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
plt.legend()
plt.title('In-data forecast with additive Holt-Winters model')
fig.savefig('in-data_forecast.png')
print('Output: in-data_forecast.png')

#----------------------------------------------------------------------------
# error metrics

# root mean squared error
def rmse(y, y_hat):
    return np.sqrt(np.mean((y - y_hat)**2))

print('Train RSME:', rmse(train[predict_col], res_holtwinters_indata.fittedvalues))
print('Test RMSE:', rmse(test[predict_col], res_holtwinters_indata.forecast(Ntest)))

#----------------------------------------------------------------------------
# instantiate Holt-Winters model for real out-data forecast
holtwinters_outdata = ExponentialSmoothing(
                      df[predict_col],
                      initialization_method='heuristic',
                      trend='add', # 'add' for linear, 'mul' for curve
                      seasonal='add', # 'add' for consant cycles, 'mul' for changing cycles
                      seasonal_periods=quarters_per_day) # cycle period is 1 day = 96 quarters

# fit model to training data
res_holtwinters_outdata = holtwinters_outdata.fit()

# future times of out-data forecast
start_date =  df.index[-1] + pd.Timedelta(freq)
future_times = pd.date_range(start=start_date, periods=Ntest, freq=freq)

# do and save out-data forecast
df_forecast = pd.DataFrame(index=future_times, columns=['Holt-Winters-out-data'])
df_forecast['Holt-Winters-out-data'] = res_holtwinters_outdata.forecast(Ntest)
print(res_holtwinters_outdata.mle_retvals)

# plot out-data forecast
ax.plot(df_forecast['Holt-Winters-out-data'], label='out-data forecast', linestyle='dashed', color='green')
plt.title('Out-data forecast with additive Holt-Winters model')
plt.legend()
fig.savefig('out-data_forecast.png')
#plt.show()
print('Output: out-data_forecast.png')
