import torch
import torch.nn as nn

# Datos de entrenamiento
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

n_samples, n_features = X.shape

# Modelo lineal
input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size)

# Pérdida y optimizador
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print(f'Predicción antes del aprendizaje: f(5) = {model(torch.tensor([5], dtype=torch.float32)).item():.3f}')

# Ciclo de entrenamiento
num_epochs = 100

for epoch in range(num_epochs):
    # Predicción
    y_pred = model(X)

    # Cálculo del error
    l = loss(Y, y_pred)

    # Cálculo del gradiente
    l.backward()

    # Optimización
    optimizer.step()

    # Resetear gradiente
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l.item():.8f}')

print(f'Predicción después del aprendizaje: f(5) = {model(torch.tensor([5], dtype=torch.float32)).item():.3f}')
