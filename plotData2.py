import pandas as pd
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates

# load photovoltaic production data
df = pd.read_csv('PV_system.csv', usecols=['Time', 'ENERGY_PRODUCED', 'POWER_PRODUCTION'], index_col='Time', parse_dates=True)

# convert W to kW
df['ENERGY_PRODUCED'] = df['ENERGY_PRODUCED'] / 1000
df['POWER_PRODUCTION'] = df['POWER_PRODUCTION'] / 1000

# time data every 15 minutes
df.index.freq = '15min'

# plot all data
fig, ax = plt.subplots(2, figsize=(9,7))

ax[0].plot(df['POWER_PRODUCTION'])
ax[0].set_ylabel('Power produced (kW)')
ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))

ax[1].plot(df['ENERGY_PRODUCED'])
ax[1].set_ylabel('Energy produced (kWh)')
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))

plt.xlabel('Date')
#fig.suptitle('Photovoltaic production (during two weeks in Oct 2020)')
plt.savefig('PV_production_allData.png')
print('Output: PV_production_allData.png')
#plt.show()

#-------------------------------------------------------------------------------
# plot data of one day only
# define which day
day = datetime.date(2020, 10, 20)

# extract data of desired day
df_day = df[df.index.date == day]

# now plot
fig, ax = plt.subplots(2, figsize=(9,7))

ax[0].plot(df_day['POWER_PRODUCTION'])
ax[0].set_ylabel('Power produced (kW)')
ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

ax[1].plot(df_day['ENERGY_PRODUCED'])
ax[1].set_ylabel('Energy produced (kWh)')
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

plt.xlabel('Time')
#fig.suptitle('Photovoltaic production on ' + day.strftime('%dth %b %Y'))
plt.savefig('PV_production_' + day.strftime('%Y-%m-%d') + '.png')
print('Output: PV_production_' + day.strftime('%Y-%m-%d') + '.png')
#plt.show()
