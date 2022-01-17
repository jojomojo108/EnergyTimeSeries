import pandas as pd
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates

# load power data
# of first few days only
quarters_per_day = int(24 * 60 / 15) # how many 15 minutes per day
days = 8 # how many days to load
nrows = days*quarters_per_day # how many data rows to load
df = pd.read_csv('PV_system.csv', usecols=['Time', 'POWER_CONSUMPTION', 'POWER_GRID', 'POWER_PRODUCTION', 'POWER_STORAGE', 'ENERGY_EXPORTED'], index_col='Time', parse_dates=True, nrows=nrows)

# time data every 15 minutes
df.index.freq = '15min'

# calculate how much of total power consumption from which power source/device
df['POWER_RATIO_GRID'] = df['POWER_GRID'] / df['POWER_CONSUMPTION'] * 100
df['POWER_RATIO_PRODUCTION'] = df['POWER_PRODUCTION'] / df['POWER_CONSUMPTION'] * 100
df['POWER_RATIO_STORAGE'] = df['POWER_STORAGE'] / df['POWER_CONSUMPTION'] * 100

# convert W to kW
df['ENERGY_EXPORTED'] = df['ENERGY_EXPORTED'] / 1000

# plot power distributions and resulting energy exportation
fig, ax = plt.subplots(2, figsize=(10,8))

ax[0].plot(df[['POWER_RATIO_GRID','POWER_RATIO_PRODUCTION','POWER_RATIO_STORAGE']])
ax[0].set_ylabel('Ratio of total consumed power (%)')
ax[0].legend(['drawn from grid', 'produced locally', 'stored in battery'], loc ="lower left")
ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
ax[0].title.set_text('Power distribution amongst devices')

# if too much power produced, then energy is exported
ax[1].plot(df['ENERGY_EXPORTED'])
ax[1].set_ylabel('Energy exported (kWh)')
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
ax[1].title.set_text('Energy exported when power overproduction')

plt.xlabel('Date')
#fig.suptitle('Power overproduction leads to energy export')
plt.savefig('Power_distribution_energy_export.png')
#plt.show()
print('Output: Power_distribution_energy_export.png')
