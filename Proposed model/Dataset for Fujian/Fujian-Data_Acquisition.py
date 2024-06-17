from datetime import datetime
from meteostat import Point, Daily, Hourly
import pandas as pd


# Set time period
start = datetime(2023, 1, 1)
end = datetime(2024, 1, 1)

# Create Point for Fujian, China
Fujian = Point(26.65, 118.1667, 128)

# Get hourly data for 2023
data = Hourly(Fujian, start, end)
data = data.fetch()
wind_speed = data.wspd[:-1]
print(wind_speed)

# Converting wind speed in km/h to m/s
wind_speed = pd.DataFrame(wind_speed*(1000/3600))
wind_speed.to_excel('Wind speed data of Fujian.xlsx', index=False)

