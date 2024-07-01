import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Leer datos
data = pd.read_csv('data.csv')
X = data.iloc[:, 0]
Y = data.iloc[:, 1]

# Mínimos cuadrados
N = len(X)
sumX = sum(X)
sumY = sum(Y)
sumXY = sum(X*Y)
sumX2 = sum(X*X)

w1 = (N*sumXY - sumX*sumY) / (N*sumX2 - sumX*sumX)
w0 = (sumY - w1*sumX) / N

# Gráfica
plt.scatter(X, Y)
plt.rcParams['figure.figsize'] = (12.0, 9.0)
plt.plot([min(X), max(X)], [min(w0 + w1*X), max(w0 + w1*X)], color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
