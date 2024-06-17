import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import RELM_class
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


sheetname = ["Sheet1", "Sheet2", "Sheet3", "Sheet4", "Sheet5", "Sheet6", "Sheet7", "Sheet8"]
all_IMF_predicted_values, all_IMF_real_values = np.zeros([1752, 3]), np.zeros([1752, 3])
writer1 = pd.ExcelWriter('Candidate features of IMF.xlsx', engine='xlsxwriter')
writer2 = pd.ExcelWriter('Selected features of IMF.xlsx', engine='xlsxwriter')
writer3 = pd.ExcelWriter('Predicted values of IMF.xlsx', engine='xlsxwriter')
writer4 = pd.ExcelWriter('Real values of IMF.xlsx', engine='xlsxwriter')
for n in range(0, 8):
    print(f'Implementing synchronization learning strategy for IMF{n + 1}')
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

        model1 = RELM_class.elm(hidden_units=int(round(solution[1], 0)), activation_function='sigmoid',
                                random_type='normal', x=x_train1, y=y_train1, C=float(solution[2]), elm_type='reg')
        model1.fit('solution2')
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


    IMFs_data = pd.read_excel(io="IMFs after MOVMD.xlsx", sheet_name="Sheet1")
    data = IMFs_data.values[0:8760, n]
    data1 = IMFs_data.values[0:7008, n]

    set_value('data1', data1)
    outlag = 3
    scalarY, x_train2, x_test2, y_train, y_test = nor_data(data, 1752, 2, 9, outlag)
    savex_test2 = scalarY.inverse_transform(x_test2)
    savex_test2 = pd.DataFrame(savex_test2)
    savex_test2.to_excel(writer1, sheet_name=sheetname[n], index=False)

    pop = 25  # Population size
    MaxIter = 30  # The max value of iterations
    dim = 3
    lb = [1, 64, 400]
    ub = [511, 1024, 3200]
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
    savex_test.to_excel(writer2, sheet_name=sheetname[n], index=False)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))

    model = RELM_class.elm(hidden_units=int(round(float(GbestPositon[1]), 0)), activation_function='sigmoid',
                           random_type='normal', x=x_train, y=y_train, C=float(GbestPositon[2]), elm_type='reg')
    model.fit('solution2')
    y_test_prediction = model.predict(x_test)
    y_test_predicted_values = scalarY.inverse_transform(y_test_prediction.reshape(-1, outlag))
    y_test_real_values = scalarY.inverse_transform(y_test.reshape(-1, outlag))

    save_y_test_predicted_values = pd.DataFrame(y_test_predicted_values)
    save_y_test_predicted_values.to_excel(writer3, sheet_name=sheetname[n], index=False)
    save_y_test_real_values = pd.DataFrame(y_test_real_values)
    save_y_test_real_values.to_excel(writer4, sheet_name=sheetname[n], index=False)

    for i in range(3):
        all_IMF_predicted_values[:, i] = all_IMF_predicted_values[:, i] + y_test_predicted_values[:, i]
        all_IMF_real_values[:, i] = all_IMF_real_values[:, i] + y_test_real_values[:, i]

writer1.save()
writer2.save()
writer3.save()
writer4.save()
print('Data saving complete')


print('Accumulating the predicted values of all subsequences and calculating the total prediction accuracy')

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



def plotcompare(predicted_values, real_values, name):
    plt.plot(predicted_values, color='blue', label='Predicted Value')
    plt.plot(real_values, color='red', label='Real Value')
    plt.title('Wind Speed Prediction')
    plt.xlabel('Time')
    plt.ylabel('Wind Speed')
    plt.legend()
    plt.savefig(name + '.jpg', dpi=600, bbox_inches='tight')
    plt.show()


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


    Predicted_trend_sequence = pd.read_excel(io="Predicted values of trend sequence.xlsx", sheet_name="Sheet1")
    sumpredicted = all_IMF_predicted_values[:, step].reshape(-1, 1) + Predicted_trend_sequence.values[:, step].reshape(-1, 1)
    save_sumpredicted = pd.DataFrame(sumpredicted)
    save_sumpredicted.to_excel('Total predicted values.xlsx', index=False)

    original_data = pd.read_excel(io="Wind speed data of Vancouver.xlsx", sheet_name="Sheet1")
    original_data = original_data.values[7006 + step:8758 + step].reshape(-1, 1)


    print(f'{step + 1}-step prediction accuracy')
    eva_test = Evaluation_indexes(sumpredicted, original_data, eva_test)

    plotcompare(sumpredicted, original_data, name=f'{step + 1}-step prediction comparison figure')

eva_test = pd.concat(
    [pd.DataFrame(['RMSE', 'MAE', 'MAPE', 'IA', 'TIC']), pd.DataFrame(eva_test)], axis=1)
eva_test.columns = ['Evaluation metrics', '1-step', '2-step', '3-step']
eva_test.to_excel('Vancouver-Evaluation metrics.xlsx', index=False)
