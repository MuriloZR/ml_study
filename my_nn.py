import numpy as np

# Remove a linearidade da rede
def relu(z):
    return np.maximum(0, z)

# Retorna 0.0 ou 1.0 dependendo do sinal do valor
def relu_grad(z):
    return (z > 0).astype(float)


# Calcula qual das saídas é a mais predominante
def softmax(z):
    z_estavel = z - np.max(0, axis=0, keepdims=True)
    e = np.exp(z_estavel)
    return e/np.sum(e, axis=0, keepdims=True)

# One-Hot: y_true é um vetor com somente UM elemento igual a 1
# A posição do elemento 1 é a classe correta
def categorical_cross_entropy(y_pred, y_true):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1-eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=0))


# Gradiente da CCE + Softmax juntos
def cce_softmax_grad(y_pred, y_true):
    return (y_pred - y_true) / y_true.shape[1]

# Inicialização de He — recomendada para ReLU
# Escala os pesos por sqrt(2/n_entrada) para manter
# a variância dos gradientes estável entre camadas
def inicializar_pesos(camadas, seed=42):
    rng = np.random.default_rng(seed)
    params = {}
    for i in range(1, len(camadas)):
        n_entrada = camadas[i-1]
        n_saida   = camadas[i]
        params[f'W{i}'] = rng.standard_normal((n_saida, n_entrada)) * np.sqrt(2 / n_entrada)
        params[f'b{i}'] = np.zeros((n_saida, 1))
    return params

# Calcula a saída da rede, com um número fixo de camadas
def forward_pass(X, params):
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]
    W3, b3 = params["W3"], params["b3"]

    Z1 = W1 @ X + b1
    A1 = relu(Z1)

    Z2 = W2 @ A1 + b2
    A2 = relu(Z2)

    Z3 = W3 @ A2 + b3
    A3 = softmax(Z3)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3, "X":X}
    return A3, cache

#calcula a saída da rede, com um número generalizado de camadas
def forward_pass_loop(X, params: dict):
    Z = []
    A = []
    cache = {}
    i = 1
    while f"W{i}" in params:
        Z.append(params[f"W{i}"] @ (X if i == 1 else A[i-2]) + params[f"b{i}"])
        A.append(relu(Z[i-1]) if f"W{i+1}" in params else softmax(Z[i-1]))
        cache[f"Z{i}"] = Z[i-1]
        cache[f"A{i}"] = A[i-1]
        i += 1

    cache["X"] = X
    return A[-1], cache

#TODO
def backward_pass(y_true, params, cache: dict):
    dZ3 = cce_softmax_grad(cache["Z3"])


# Nossa arquitetura: 784 → 128 → 64 → 10
camadas = [784, 128, 64, 10]
params = inicializar_pesos(camadas)

# Verificando as dimensões
for nome, arr in params.items():
    print(f"{nome}: {arr.shape}")

# Dados fictícios: 784 features, 5 amostras
X_teste = np.random.randn(784, 5)

A3_direto, cache_direto = forward_pass(X_teste, params)
A3_loop,   cache_loop   = forward_pass_loop(X_teste, params)

print("Saídas iguais?", np.allclose(A3_direto, A3_loop))
print("Shape da saída:", A3_direto.shape)  # esperado: (10, 5)
print("Soma das probabilidades:", A3_direto.sum(axis=0).round(4))  # esperado: tudo 1.0
