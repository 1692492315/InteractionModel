import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def nor_data(df, test_size, dim, in_lag, out_lag):
    train, test = df[:-test_size], df[-test_size:]
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_sc = scaler.fit_transform(train.reshape(-1, 1))
    test_sc = scaler.transform(test.reshape(-1, 1))
    data_sc = pd.concat([pd.DataFrame(train_sc), pd.DataFrame(test_sc)], axis=0).values

    if dim == 2:
        data_sc = data_sc.reshape(-1, )
    else:
        data_sc = data_sc.reshape(-1, data_sc.shape[1])
    x, y = split_data(data_sc, in_lag=in_lag, out_lag=out_lag)
    train_sc_x, test_sc_x, train_sc_y, test_sc_y = train_test_split(x, y.reshape(-1, out_lag), test_size=test_size, shuffle=False)
    return scaler, train_sc_x, test_sc_x, train_sc_y, test_sc_y


def split_data(df, in_start=0, in_lag=None, out_lag=None):
    if isinstance(df, pd.DataFrame):
        df = np.array(df)
    datax, datay = [], []
    for i in range(len(df)):
        in_end = in_start + in_lag
        out_end = in_end + out_lag
        if out_end < len(df) + 1:
            a = df[in_start:in_end]
            if isinstance(df, pd.Series):
                b = df[in_end:out_end]
            else:
                b = df[in_end:out_end].reshape(-1, )
            datax.append(a)
            datay.append(b)
        in_start += 1
    datax, datay = np.array(datax), np.array(datay)
    return datax, datay
