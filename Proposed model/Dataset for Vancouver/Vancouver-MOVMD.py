from platypus import GDE3, Problem, Real, nondominated, unique
from vmdpy import VMD
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
import EntropyHub as Eh
import matplotlib.pyplot as plt


tf.random.set_seed(8)


def vmd_decom(series, name, alpha, K, draw=1, isrun=1):
    if isrun:
        print('##################################')
        print('Decomposition is running')
        alpha = alpha
        tau = 0
        K = K
        DC = 0
        init = 1
        tol = 1e-7
        u, u_hat, omega = VMD(series, alpha, tau, K, DC, init, tol)
        if draw:
            # Plot original data
            fig = plt.figure(figsize=(16, 2*K))
            plt.subplot(1+K, 1, 1)
            plt.plot(series)
            plt.ylabel('Original')
            # Plot IMFs
            for i in range(K):
                plt.subplot(1+K, 1, i+2)
                plt.plot(u[i, :], linewidth=0.2, c='r')
                plt.ylabel('IMF{}'.format(i + 1))
            fig.align_labels()
            plt.tight_layout()
            plt.savefig(name + '.tif', dpi=600, bbox_inches='tight')
            plt.draw()
            plt.pause(3)
            plt.close()
        vmd_df = pd.DataFrame(u.T)
        vmd_df.columns = ['imf1-' + str(i+1) for i in range(K)]
        pd.DataFrame.to_excel(vmd_df, name + '.xlsx', index=False)
        print('Decomposition complete')
    else:
        vmd_df = pd.read_excel(name + '.xlsx')

    return vmd_df


def multi_objective_fitness(vars):
    data = pd.read_excel(io="remaining_sequence.xlsx", sheet_name="Sheet1")
    data = data.values

    alpha = round(vars[0])  # Bandwidth limitation
    tau = 0
    K = round(vars[1])  # Number of IMF
    DC = 0
    init = 1
    tol = 1e-7
    u, u_hat, omega = VMD(data, alpha, tau, K, DC, init, tol)
    u1 = u.T

    SUMK = np.zeros([8760, 1])
    for i in range(K):
        SUMK = SUMK + u1[:, i].reshape(-1, 1)

    SUMFE = 0
    for k in range(K):
        series = u1[:, k]
        FE, _, _ = Eh.FuzzEn(series, m=2)
        SUMFE = SUMFE + FE[1]

    f1 = mean_absolute_error(data, SUMK)*round(vars[1])
    f2 = SUMFE/round(vars[1])

    return f1, f2


# Setting the parameters of Multi-objective differential evolution
problem = Problem(2, 2)
problem.types[:] = [Real(100, 1000), Real(2, 10)]
problem.function = multi_objective_fitness
algorithm = GDE3(problem, population_size=50)
algorithm.run(50)

list_var = []
list_obj = []
for solution in unique(nondominated(algorithm.result)):
    list_var.append(solution.variables)
    list_obj.append(list(solution.objectives))

    print('Pareto solution', solution.variables)
    print('Corresponding objective function values', solution.objectives)


df_result = pd.concat([pd.DataFrame(list_var), pd.DataFrame(list_obj)], axis=1)
df_result.columns = ['alpha', 'K', 'f1', 'f2']
df_result_sort = df_result.sort_values(by='f1', axis=0, ascending=True)
print(df_result_sort)

# The higher preference for the relative importance of the objective function f1
print('min f1', df_result_sort.iloc[0, :])
best_alpha = df_result_sort.iloc[0, 0]
best_K = df_result_sort.iloc[0, 1]
best_f1 = df_result_sort.iloc[0, 2]
best_f2 = df_result_sort.iloc[0, 3]

data = pd.read_excel(io="remaining_sequence.xlsx", sheet_name="Sheet1")
data = data.values
vmd_decom(data, 'IMFs after MOVMD', round(best_alpha), round(best_K), draw=1, isrun=1)