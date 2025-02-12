import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit

# Leer datos
data = pd.read_csv('data.csv')
X = np.array(data.iloc[:, 0])
Y = np.array(data.iloc[:, 1])

# Mínimos cuadrados
N = len(X)
sumX = np.sum(X)
sumY = np.sum(Y)
sumXY = np.sum(X*Y)
sumX2 = np.sum(X*X)

w1 = (N*sumXY - sumX*sumY) / (N*sumX2 - sumX*sumX)
w0 = (sumY - w1*sumX) / N
Ybar = w0 + w1*X

# Descenso de gradiente
w0 = 0.0
w1 = 0.0
alpha = 0.025
epocs = 100

@jit(nopython=True)
def descensoG(epocs, sumX, sumY, sumXY, sumX2, N, alpha):
    w0 = 0.0
    w1 = 0.0
    for i in range(epocs):
        Gradw0 = -2.0*(sumY - w0*N - w1*sumX)
        Gradw1 = -2.0*(sumXY - w0*sumX - w1*sumX2)
        w0 -= alpha*Gradw0
        w1 -= alpha*Gradw1
    return w0, w1

w0, w1 = descensoG(epocs, sumX, sumY, sumXY, sumX2, N, alpha)
Ybar2 = w0 + w1*X

# Gráfica
plt.scatter(X, Y)
plt.rcParams['figure.figsize'] = (12.0, 9.0)
plt.plot([min(X), max(X)], [min(Ybar), max(Ybar)], color='red')
plt.plot([min(X), max(X)], [min(Ybar2), max(Ybar2)], color='green')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
