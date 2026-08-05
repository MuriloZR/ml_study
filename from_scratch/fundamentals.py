import numpy as np
# Entradas de um neurônio (3 features)
x = np.array([1.0, 2.0, 3.0])
# Pesos aprendidos
w = np.array([0.5, -0.3, 0.8])
# Bias
b = 0.1
# Saída do neurônio (antes da ativação)
z = np.dot(x, w) + b
print(z) # 2.0

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Camada com 4 neurônios
# W: shape (4, 3) — 4 neurônios, 3 entradas cada
W1 = np.random.randn(4, 3) * 0.1
b1 = np.zeros(4)
# Forward pass pela primeira camada
z1 = W1 @ x + b1
# shape (4,)
a1 = relu(z1)
# ativação
# Camada de saída (1 neurônio)
W2 = np.random.randn(1, 4) * 0.1
b2 = np.zeros(1)
z2 = W2 @ a1 + b2
output = sigmoid(z2) # probabilidade entre 0 e 1

print(output)

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

# Gradiente da MSE em relação à saída
def mse_grad(y_pred, y_true):
    return 2 * (y_pred - y_true) / len(y_true)
# Backprop pela camada de saída
y_true = np.array([1.0])
loss = mse_loss(output, y_true)
# dL/dz2 = gradiente da loss * derivada do sigmoid
dL_dz2 = mse_grad(output, y_true) * (output * (1 - output))
# Gradientes dos pesos da camada 2
dL_dW2 = dL_dz2[:, None] * a1[None, :]
dL_db2 = dL_dz2
print(dL_db2)