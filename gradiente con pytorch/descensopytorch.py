import torch

# Tensor a optimizar -> requires_grad=True
w = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(1.0)
y = torch.tensor(2.0)

# Evaluación cálculo de costo
y_predicted = w * x
loss = (y_predicted - y) ** 2
print(loss)

# Retropropagación para calcular gradiente
loss.backward()
print(w.grad)

# Nuevos coeficientes
# Repetir evaluación y retropropagación
with torch.no_grad():
    w -= 0.01 * w.grad
    w.grad.zero_()

print(w)
