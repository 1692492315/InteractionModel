import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import xgboost as xgb
from Dataset_Division import nor_data
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


tf.random.set_seed(8)


# Define global variable function
def _init():
    global _global_dict
    _global_dict = {}

def set_value(key, value):
    _global_dict[key] = value

def get_value(key):
    try:
        return _global_dict[key]
    except:
        print('read'+key+'failure\r\n')


_init()


def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index


def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[1]):
        Xnew[:, i] = X[:, index[i]].reshape(-1, )
    return Xnew


def hyperopt_objective(params):
    data1 = get_value('data1')
    outlag1 = 3
    scalarY1, x_train0, x_val0, y_train0, y_val = nor_data(data1, 1752, 2, 9, outlag1)
    pick1 = get_value('pick')

    x_train1 = []
    x_val = []
    for i in range(9):
        if pick1[i] == 1:
            x_train1.append(x_train0[:, i])
            x_val.append(x_val0[:, i])
        else:
            pass
    x_train1 = np.array(x_train1).T
    x_val = np.array(x_val).T

    x_train1 = np.reshape(x_train1, (x_train1.shape[0], x_train1.shape[1], 1))
    y_train1 = y_train0
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))


    model1 = Sequential()
    model1.add(Conv1D(filters=int(params["filters"]), kernel_size=int(params["kernel_size"]), padding='causal',
                      dilation_rate=int(params["dilation_rate"]), input_shape=(x_train.shape[1], x_train.shape[2])))
    model1.add(MaxPooling1D(2))
    model1.add(Conv1D(filters=2*int(params["filters"]), kernel_size=int(params["kernel_size"]), padding='causal',
                      dilation_rate=int(params["dilation_rate"])))
    model1.add(Flatten())
    model1.add(Dense(3))
    model1.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error')
    model1.fit(x_train1, y_train1, batch_size=32, epochs=100, verbose=1)
    prediction1 = model1.predict(x_val)

    predicted_values1 = scalarY1.inverse_transform(prediction1.reshape(-1, outlag1))
    real_values1 = scalarY1.inverse_transform(y_val.reshape(-1, outlag1))
    predicted_values1 = predicted_values1.ravel()
    real_values1 = real_values1.ravel()
    predicted_values1 = predicted_values1.reshape(-1, 1)
    real_values1 = real_values1.reshape(-1, 1)

    rmse1 = math.sqrt(mean_squared_error(predicted_values1, real_values1))
    mae1 = mean_absolute_error(predicted_values1, real_values1)
    mape1 = np.mean(np.abs((predicted_values1 - real_values1) / real_values1))
    MCS = 1 / 3 * rmse1 + 1 / 3 * mae1 + 1 / 3 * mape1
    return MCS


original_data = pd.read_excel(io="Wind speed data of Fujian.xlsx", sheet_name="Sheet1")
data = original_data.values[0:8760]
data1 = original_data.values[0:7008]

set_value('data1', data1)
outlag = 3
scalarY, x_train2, x_test2, y_train, y_test = nor_data(data, 1752, 2, 9, outlag)


# Feature selection using XGB
model2 = xgb.XGBRegressor()
model2.fit(x_train2, y_train[:, 0])
feature_importances = model2.feature_importances_
fitness, sortIndex = SortFitness(feature_importances)
Feature1 = SortPosition(x_train2, sortIndex)
Feature = Feature1[:, 4:9]

pick = np.zeros([9, 1])
for i in range(9):
    for j in range(5):
        if x_train2[:, i].tolist() == Feature[:, j].tolist():
            pick[i] = 1
        else:
            pass

set_value('pick', pick)
print(pick.T)

x_train = []
x_test = []
for i in range(9):
    if pick[i] == 1:
        x_train.append(x_train2[:, i])
        x_test.append(x_test2[:, i])
    else:
        pass
x_train = np.array(x_train).T
x_test = np.array(x_test).T

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# Defining the parameters space
param_grid_simple = {
    'filters': hp.quniform("filters", 8, 100, 1),
    'kernel_size': hp.quniform("kernel_size", 1, 10, 1),
    'dilation_rate': hp.quniform("dilation_rate", 1, 10, 1)
}

# Initializing the Trials object
trials = Trials()

# Running TPE optimization
params_best = fmin(hyperopt_objective, space=param_grid_simple, algo=tpe.suggest, max_evals=260, trials=trials)
filters_value = params_best['filters']
kernel_size_value = params_best['kernel_size']
dilation_rate_value = params_best['dilation_rate']


# Obtaining the optimal parameters
print("optimal parameters：", params_best)
print("optimal filters：", filters_value)
print("optimal kernel size：", kernel_size_value)
print("optimal dilation rate：", dilation_rate_value)


# Constructing model with the optimal parameters
model = Sequential()
model.add(Conv1D(filters=int(filters_value), kernel_size=int(kernel_size_value), padding='causal',
                 dilation_rate=int(dilation_rate_value), input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(MaxPooling1D(2))
model.add(Conv1D(filters=2*int(filters_value), kernel_size=int(kernel_size_value), padding='causal',
                 dilation_rate=int(dilation_rate_value)))
model.add(Flatten())
model.add(Dense(3))
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=1)
y_test_prediction = model.predict(x_test)
y_test_predicted_values = scalarY.inverse_transform(y_test_prediction.reshape(-1, outlag))
y_test_real_values = scalarY.inverse_transform(y_test.reshape(-1, outlag))

save_y_test_predicted_values = pd.DataFrame(y_test_predicted_values)
save_y_test_predicted_values.to_excel('Predicted values of Fujian-XGB-TCN-TPE.xlsx', index=False)
save_y_test_real_values = pd.DataFrame(y_test_real_values)
save_y_test_real_values.to_excel('Real values of Fujian-XGB-TCN-TPE.xlsx', index=False)


def ia(y: np.array, y_pre: np.array) -> float:
    """
    :param y: the actual values
    :param y_pre: the prediction values
    :return: Index of agreement (IA)
    """
    arr_y = y.ravel()
    arr_y_pre = y_pre.ravel()
    mean_y = arr_y.mean()

    arr_1 = np.square(arr_y - arr_y_pre)
    arr_2 = np.square(np.abs(arr_y_pre - mean_y) + np.abs(arr_y - mean_y))

    IA = 1 - (arr_1.sum() / arr_2.sum())
    return IA


def tic(y: np.array, y_pre: np.array) -> float:
    """
    :param y: the actual values
    :param y_pre: the prediction values
    :return: Theil's inequality coefficient (TIC)
    """
    arr_y = y.ravel()
    arr_y_pre = y_pre.ravel()

    TIC = math.sqrt(np.square(arr_y_pre - arr_y).mean())/(math.sqrt(np.square(arr_y).mean()) + math.sqrt(np.square(arr_y_pre).mean()))

    return TIC


eva_test = np.zeros((5, 3))
for step in range(3):
    def Evaluation_indexes(predicted_values, real_values, eva_test):
        rmse = math.sqrt(mean_squared_error(real_values, predicted_values))
        mae = mean_absolute_error(real_values, predicted_values)
        mape = np.mean(np.abs((predicted_values - real_values) / real_values))
        IA = ia(real_values, predicted_values)
        TIC = tic(real_values, predicted_values)
        print('RMSE: %.4f' % rmse)
        print('MAE: %.4f' % mae)
        print('MAPE: %.6f' % mape)
        print('IA: %.4f' % IA)
        print('TIC: %.4f' % TIC)
        eva_test[0, step] = round(rmse, 4)
        eva_test[1, step] = round(mae, 4)
        eva_test[2, step] = round(float(mape), 4)
        eva_test[3, step] = round(IA, 4)
        eva_test[4, step] = round(TIC, 4)
        return eva_test


    sumpredicted = y_test_predicted_values[:, step]
    sumreal = y_test_real_values[:, step]

    print(f'{step + 1}-step prediction accuracy')
    eva_test = Evaluation_indexes(sumpredicted, sumreal, eva_test)

eva_test = pd.concat(
    [pd.DataFrame(['RMSE', 'MAE', 'MAPE', 'IA', 'TIC']), pd.DataFrame(eva_test)], axis=1)
eva_test.columns = ['Evaluation metrics', '1-step', '2-step', '3-step']
eva_test.to_excel('Fujian-XGB-TCN-TPE.xlsx', index=False)

