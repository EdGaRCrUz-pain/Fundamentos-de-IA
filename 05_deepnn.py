import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numba import jit

# Leer imágenes
data = pd.read_csv('train.csv')

# Convertir a numpy y revolver imágenes
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

# Separar conjunto de datos
# dev: prueba
# test: entrenamiento
data_dev = data[0:1000].T
Y_dev = data_dev[0].astype(int)
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0].astype(int)
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape

# Derivada de la ReLU
def ReLU_deriv(Z):
    return Z > 0

# Codificación de la clasificación
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

# Cálculo numérico del gradiente
def backward_prop(Z, A, W, X, Y):
    dW = []
    db = []
    m = X.shape[1]
    n = len(W) - 1
    one_hot_Y = one_hot(Y)
    dZ = A[n] - one_hot_Y
    dW.append(1 / m * dZ.dot(A[n-1].T))
    db.append(1 / m * np.sum(dZ, axis=1, keepdims=True))
    for i in range(n - 1, 0, -1):
        dZ = W[i + 1].T.dot(dZ) * ReLU_deriv(Z[i])
        dW.append(1 / m * dZ.dot(A[i-1].T))
        db.append(1 / m * np.sum(dZ, axis=1, keepdims=True))
    dZ = W[1].T.dot(dZ) * ReLU_deriv(Z[0])
    dW.append(1 / m * dZ.dot(X.T))
    db.append(1 / m * np.sum(dZ, axis=1, keepdims=True))
    dW.reverse()
    db.reverse()
    return dW, db

# Mejorar parámetros
def update_params(W, b, dW, db, alpha):
    for i in range(len(W)):
        W[i] = W[i] - alpha * dW[i]
        b[i] = b[i] - alpha * db[i]
    return W, b

# Predicciones
def get_predictions(A2):
    return np.argmax(A2, axis=0)

# Precisión
def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# Inicialización de parámetros
def init_params():
    W1 = np.random.randn(10, 784) - 0.5
    b1 = np.random.randn(10, 1) - 0.5
    W2 = np.random.randn(10, 10) - 0.5
    b2 = np.random.randn(10, 1) - 0.5
    return [W1, W2], [b1, b2]

# Función de activación ReLU
def ReLU(Z):
    return np.maximum(Z, 0)

# Función de activación softmax
def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A

# Evaluar red
def forward_prop(W, b, X):
    Z1 = W[0].dot(X) + b[0]
    A1 = ReLU(Z1)
    Z2 = W[1].dot(A1) + b[1]
    A2 = softmax(Z2)
    return [Z1, Z2], [A1, A2]

# Descenso de gradiente
def gradient_descent(X, Y, alpha, iterations):
    W, b = init_params()
    for i in range(iterations):
        Z, A = forward_prop(W, b, X)
        dW, db = backward_prop(Z, A, W, X, Y)
        W, b = update_params(W, b, dW, db, alpha)
        if i % 10 == 0:
            print("Iteración:", i)
            predictions = get_predictions(A[-1])
            print("Precisión:", get_accuracy(predictions, Y))
    return W, b

# Entrenar la red
W, b = gradient_descent(X_train, Y_train, 0.10, 1000)

# Hacer predicciones
def make_predictions(X, W, b):
    _, A = forward_prop(W, b, X)
    predictions = get_predictions(A[-1])
    return predictions

# Evaluar predicciones
def test_prediction(index, W, b):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W, b)
    label = Y_train[index]
    print("Prediction:", prediction)
    print("Label:", label)
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

for i in range(20):
    test_prediction(i, W, b)
