import pandas as pd
import matplotlib.pyplot as plt


def double_exponential_smoothing(series, alpha, beta):
    """
        series - dataset with timeseries
        alpha - float [0.0, 1.0], smoothing parameter for level
        beta - float [0.0, 1.0], smoothing parameter for trend
    """
    # first value is same as series
    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): # forecasting
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)
    return result


original_data = pd.read_excel(io="Wind speed data of Fujian.xlsx", sheet_name="Sheet1")
original_data = original_data.values[0:8760]

alpha = 0.2
beta = 0.2
trend_sequence = double_exponential_smoothing(original_data, alpha, beta)
remaining_sequence = original_data - trend_sequence[0:8760]

trend_sequence = pd.DataFrame(trend_sequence[0:8760])
trend_sequence.to_excel('trend_sequence.xlsx', index=False)

remaining_sequence = pd.DataFrame(remaining_sequence[0:8760])
remaining_sequence.to_excel('remaining_sequence.xlsx', index=False)

plt.plot(original_data, color='red', label='original_data')
plt.plot(trend_sequence, color='green', label='trend_sequence')
plt.plot(remaining_sequence, color='blue', label='remaining_sequence')
plt.title('double_exponential_smoothing')
plt.xlabel('Time')
plt.ylabel('Wind Speed')
plt.legend()
plt.show()