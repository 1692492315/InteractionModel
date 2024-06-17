import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import xgboost as xgb
from Dataset_Division import nor_data
import IBES


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


# Decimal to binary
def DecToBin(i):
    list = []
    while i:
        list.append(i % 2)
        i = i // 2
    list.reverse()
    return list


def fitness_function(solution):
    data1 = get_value('data1')
    outlag1 = 3
    scalarY1, x_train0, x_val0, y_train0, y_val = nor_data(data1, 1752, 2, 9, outlag1)

    Binlist = DecToBin(int(round(solution[0], 0)))
    x_train1 = []
    x_val = []
    for i in range(len(Binlist)):
        if Binlist[i] == 1:
            x_train1.append(x_train0[:, (9 - len(Binlist))+i])
            x_val.append(x_val0[:, (9 - len(Binlist)) + i])
        else:
            pass
    x_train1 = np.array(x_train1).T
    x_val = np.array(x_val).T

    x_train1 = np.reshape(x_train1, (x_train1.shape[0], x_train1.shape[1]))
    y_train1 = y_train0
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1]))

    model1 = xgb.XGBRegressor(learning_rate=float(solution[1]), n_estimators=int(round(solution[2], 0)))
    model1.fit(x_train1, y_train1)
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
    MCS = 1/3*rmse1+1/3*mae1+1/3*mape1
    return MCS


trend_sequence = pd.read_excel(io="trend_sequence.xlsx", sheet_name="Sheet1")
data = trend_sequence.values
data1 = trend_sequence.values[0:7008]

set_value('data1', data1)
outlag = 3
scalarY, x_train2, x_test2, y_train, y_test = nor_data(data, 1752, 2, 9, outlag)
savex_test2 = scalarY.inverse_transform(x_test2)
savex_test2 = pd.DataFrame(savex_test2)
savex_test2.to_excel('Candidate features of trend sequence.xlsx', index=False)

pop = 25   # Population size
MaxIter = 30   # The max value of iterations
dim = 3
lb = [1, 0.01, 1]
ub = [511, 1, 100]
fobj = fitness_function
GbestScore, GbestPositon1, Curve = IBES.IBES(pop, dim, lb, ub, MaxIter, fobj)
GbestPositon = GbestPositon1.T
print('Optimal solution：', GbestPositon)
print('Optimal fitness value：', GbestScore)

Binlist1 = DecToBin(int(round(float(GbestPositon[0]), 0)))
print('Binary code corresponding to the selected features：', Binlist1)

x_train = []
x_test = []
for i in range(len(Binlist1)):
    if Binlist1[i] == 1:
        x_train.append(x_train2[:, (9 - len(Binlist1)) + i])
        x_test.append(x_test2[:, (9 - len(Binlist1)) + i])
    else:
        pass
x_train = np.array(x_train).T
x_test = np.array(x_test).T

savex_test = scalarY.inverse_transform(x_test)
savex_test = pd.DataFrame(savex_test)
savex_test.to_excel('Selected features of trend sequence.xlsx', index=False)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))

model = xgb.XGBRegressor(learning_rate=float(GbestPositon[1]), n_estimators=int(round(float(GbestPositon[2]), 0)))
model.fit(x_train, y_train)
y_test_prediction = model.predict(x_test)
y_test_predicted_values = scalarY.inverse_transform(y_test_prediction.reshape(-1, outlag))
y_test_real_values = scalarY.inverse_transform(y_test.reshape(-1, outlag))

save_y_test_predicted_values = pd.DataFrame(y_test_predicted_values)
save_y_test_predicted_values.to_excel('Predicted values of trend sequence.xlsx', index=False)
save_y_test_real_values = pd.DataFrame(y_test_real_values)
save_y_test_real_values.to_excel('Real values of trend sequence.xlsx', index=False)


for i in range(outlag):
    print(f'{i + 1}-step prediction accuracy')
    predicted_values = y_test_predicted_values[:, i]
    real_values = y_test_real_values[:, i]

    predicted_values = predicted_values.reshape(-1, 1)
    real_values = real_values.reshape(-1, 1)

    rmse = math.sqrt(mean_squared_error(predicted_values, real_values))
    mae = mean_absolute_error(predicted_values, real_values)
    mape = np.mean(np.abs((predicted_values - real_values) / real_values))
    R2 = r2_score(real_values, predicted_values)

    print('RMSE: %.4f' % rmse)
    print('MAE: %.4f' % mae)
    print('MAPE: %.4f' % mape)