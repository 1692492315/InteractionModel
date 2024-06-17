import numpy as np
import random
import copy
from pyDOE import lhs


''' Population initialization by Latin hypercube sampling'''
def initial(pop, dim, lb, ub):
    X = []
    for i in range(dim):
        X1 = lb[i] + (ub[i] - lb[i]) * lhs(1, pop)
        X.append(X1)
    X = np.array(X).T
    X = X.reshape(pop, dim)
    return X


'''Boundary checking function'''
def BorderCheck(X, ub, lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]
            elif X[i, j] < lb[j]:
                X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]
    return X


'''Function for calculating fitness'''
def CaculateFitness(X, fun):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness


''' Improved bald eagle search algorithm'''
def IBES(pop, dim, lb, ub, MaxIter, fun):
    X = initial(pop, dim, lb, ub)
    fitness = CaculateFitness(X, fun)
    minIndex = np.argmin(fitness)
    Xnew = copy.copy(X)
    GbestScore = copy.copy(fitness[minIndex])
    GbestPositon = np.zeros([1, dim])
    GbestPositon[0, :] = copy.copy(X[minIndex, :])
    Curve = np.zeros([MaxIter, 1])
    for t in range(MaxIter):
        print(str(t+1) + "iteration")

        # 1.Selecting search space
        lm = 2.0
        Mean = np.mean(X)
        for i in range(pop):
            Xnew[i, :] = GbestPositon + lm * np.random.random() * (Mean - X[i, :])
        Xnew = BorderCheck(Xnew, ub, lb, pop, dim)
        fitnessNew = CaculateFitness(Xnew, fun)
        for i in range(pop):
            if fitnessNew[i] < fitness[i]:
                fitness[i] = copy.copy(fitnessNew[i])
                X[i, :] = copy.copy(Xnew[i, :])
        minIndex = np.argmin(fitness)
        if fitness[minIndex] < GbestScore:
            GbestScore = copy.copy(fitness[minIndex])
            GbestPositon[0, :] = copy.copy(X[minIndex, :])

        # 2.Searching space prey
        Mean = np.mean(X)
        a = 10
        R = 1.5
        for i in range(pop - 1):
            th = a * np.pi * np.random.random([pop, 1])
            r = th + R * np.random.random([pop, 1])
            xR = r * np.sin(th)
            yR = r * np.cos(th)
            x = xR / np.max(np.abs(xR))
            y = yR / np.max(np.abs(yR))
            Xnew[i, :] = X[i, :] + y[i] * (X[i, :] - X[i + 1, :]) + x[i] * (X[i, :] - Mean)
        for i in range(pop):
            if fitnessNew[i] < fitness[i]:
                fitness[i] = copy.copy(fitnessNew[i])
                X[i, :] = copy.copy(Xnew[i, :])
        minIndex = np.argmin(fitness)
        if fitness[minIndex] < GbestScore:
            GbestScore = copy.copy(fitness[minIndex])
            GbestPositon[0, :] = copy.copy(X[minIndex, :])

        # 3.Swooping down to capture prey
        Mean = np.mean(X)
        a = 10
        R = 1.5
        for i in range(pop):
            th = a * np.pi * np.random.random([pop, 1])
            r = th
            xR = r * np.sinh(th)
            yR = r * np.cosh(th)
            x = xR / np.max(np.abs(xR))
            y = yR / np.max(np.abs(yR))
            # Cauchy mutation position updating
            Xnew[i, :] = (np.random.random() * GbestPositon + x[i] * (X[i, :] - 2.0 * Mean) +
                          y[i] * (X[i, :] - 2.0 * GbestPositon)) * (1 + np.random.standard_cauchy())
        for i in range(pop):
            if fitnessNew[i] < fitness[i]:
                fitness[i] = copy.copy(fitnessNew[i])
                X[i, :] = copy.copy(Xnew[i, :])
        minIndex = np.argmin(fitness)
        if fitness[minIndex] < GbestScore:
            GbestScore = copy.copy(fitness[minIndex])
            GbestPositon[0, :] = copy.copy(X[minIndex, :])

        Curve[t] = GbestScore

    return GbestScore, GbestPositon, Curve
