"""
Rede Neural do Zero — Classificação Binária
============================================
Arquitetura: 2 entradas → camada oculta (4 neurônios, ReLU) → saída (1 neurônio, Sigmoid)
Problema: classificar pontos (x1, x2) em duas classes (0 ou 1)
"""

import numpy as np

# ─────────────────────────────────────────────
# 1. FUNÇÕES DE ATIVAÇÃO E SUAS DERIVADAS
# ─────────────────────────────────────────────

def relu(z):
    """ReLU: max(0, z) — usada na camada oculta"""
    return np.maximum(0, z)

def relu_grad(z):
    """Derivada da ReLU: 1 se z > 0, senão 0"""
    return (z > 0).astype(float)

def sigmoid(z):
    """Sigmoid: 1/(1+e^-z) — usada na saída para probabilidade"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_grad(z):
    """Derivada da Sigmoid: s(z) * (1 - s(z))"""
    s = sigmoid(z)
    return s * (1 - s)


# ─────────────────────────────────────────────
# 2. FUNÇÃO DE PERDA (BINARY CROSS-ENTROPY)
# ─────────────────────────────────────────────

def binary_cross_entropy(y_pred, y_true):
    """
    Mede quão erradas são as previsões.
    Penaliza mais quando o modelo está confiante e errado.
    """
    eps = 1e-15  # evita log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_grad(y_pred, y_true):
    """Gradiente da BCE em relação à saída do modelo"""
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / len(y_true)


# ─────────────────────────────────────────────
# 3. INICIALIZAÇÃO DOS PESOS
# ─────────────────────────────────────────────

def inicializar_pesos(n_entrada, n_oculta, n_saida, seed=42):
    """
    Inicialização de He — recomendada para ReLU.
    Escala os pesos por sqrt(2/n) para evitar gradientes que somem ou explodem.
    """
    rng = np.random.default_rng(seed)
    params = {
        # Camada 1: entrada → oculta
        "W1": rng.standard_normal((n_oculta, n_entrada)) * np.sqrt(2 / n_entrada),
        "b1": np.zeros((n_oculta, 1)),

        # Camada 2: oculta → saída
        "W2": rng.standard_normal((n_saida, n_oculta)) * np.sqrt(2 / n_oculta),
        "b2": np.zeros((n_saida, 1)),
    }
    return params


# ─────────────────────────────────────────────
# 4. FORWARD PASS — dados fluem da entrada à saída
# ─────────────────────────────────────────────

def forward_pass(X, params):
    """
    Propaga os dados pela rede camada por camada.
    Guarda os valores intermediários (cache) para o backprop.

    X: shape (n_features, n_amostras)
    Retorna: previsões e cache com valores intermediários
    """
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]

    # Camada oculta
    Z1 = W1 @ X + b1        # combinação linear
    A1 = relu(Z1)            # ativação ReLU

    # Camada de saída
    Z2 = W2 @ A1 + b2       # combinação linear
    A2 = sigmoid(Z2)         # ativação Sigmoid → probabilidade

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "X": X}
    return A2, cache


# ─────────────────────────────────────────────
# 5. BACKWARD PASS — calcula gradientes pela regra da cadeia
# ─────────────────────────────────────────────

def backward_pass(y_true, params, cache):
    """
    Propaga o erro de volta pela rede usando a regra da cadeia.
    Calcula dL/dW e dL/db para cada camada.

    Regra da cadeia: dL/dW1 = dL/dA2 * dA2/dZ2 * dZ2/dA1 * dA1/dZ1 * dZ1/dW1
    """
    m = y_true.shape[1]  # número de amostras
    W2 = params["W2"]   
    Z1, A1, Z2, A2, X = cache["Z1"], cache["A1"], cache["Z2"], cache["A2"], cache["X"]

    # ── Camada de saída (camada 2) ──
    # Gradiente da loss em relação à saída
    dA2 = binary_cross_entropy_grad(A2, y_true)

    # Gradiente de A2 em relação a Z2 (derivada do sigmoid)
    dZ2 = dA2 * sigmoid_grad(Z2)

    # Gradientes dos pesos da camada 2
    dW2 = dZ2 @ A1.T / m
    db2 = np.mean(dZ2, axis=1, keepdims=True)

    # ── Camada oculta (camada 1) ──
    # Propaga o gradiente para trás pela camada 2
    dA1 = W2.T @ dZ2

    # Gradiente de A1 em relação a Z1 (derivada do ReLU)
    dZ1 = dA1 * relu_grad(Z1)

    # Gradientes dos pesos da camada 1
    dW1 = dZ1 @ X.T / m
    db1 = np.mean(dZ1, axis=1, keepdims=True)

    gradientes = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return gradientes


# ─────────────────────────────────────────────
# 6. ATUALIZAÇÃO DOS PESOS (GRADIENT DESCENT)
# ─────────────────────────────────────────────

def atualizar_pesos(params, gradientes, lr):
    """
    Ajusta cada peso na direção que reduz a loss.
    Regra: w = w - lr * dL/dw
    """
    params["W1"] -= lr * gradientes["dW1"]
    params["b1"] -= lr * gradientes["db1"]
    params["W2"] -= lr * gradientes["dW2"]
    params["b2"] -= lr * gradientes["db2"]
    return params


# ─────────────────────────────────────────────
# 7. LOOP DE TREINAMENTO COMPLETO
# ─────────────────────────────────────────────

def treinar(X, y, n_oculta=4, lr=0.1, epochs=1000, verbose=True):
    """
    Treina a rede neural por um número de épocas.

    X: features, shape (n_features, n_amostras)
    y: labels, shape (1, n_amostras)
    """
    n_entrada = X.shape[0]
    n_saida = 1
    params = inicializar_pesos(n_entrada, n_oculta, n_saida)
    historico_loss = []

    for epoch in range(epochs):
        # Passo 1: forward — calcular previsões
        y_pred, cache = forward_pass(X, params)

        # Passo 2: calcular loss (erro)
        loss = binary_cross_entropy(y_pred, y)
        historico_loss.append(loss)

        # Passo 3: backward — calcular gradientes
        gradientes = backward_pass(y, params, cache)

        # Passo 4: atualizar pesos
        params = atualizar_pesos(params, gradientes, lr)

        # Imprimir progresso
        if verbose and epoch % 500 == 0:
            y_classe = (y_pred >= 0.5).astype(int)
            acuracia = np.mean(y_classe == y) * 100
            print(f"Época {epoch:4d} | Loss: {loss:.4f} | Acurácia: {acuracia:.1f}%")

    return params, historico_loss


def prever(X, params):
    """Retorna probabilidades e classes previstas (0 ou 1)"""
    probabilidades, _ = forward_pass(X, params)
    classes = (probabilidades >= 0.5).astype(int)
    return probabilidades, classes


# ─────────────────────────────────────────────
# 8. DADOS DE EXEMPLO — problema XOR
# ─────────────────────────────────────────────
#
# XOR é um problema clássico que redes lineares NÃO conseguem resolver.
# Demonstra por que precisamos de camadas ocultas com ativações não-lineares.
#
#  x1=0, x2=0 → 0    x1=0, x2=1 → 1
#  x1=1, x2=0 → 1    x1=1, x2=1 → 0

if __name__ == "__main__":
    print("=" * 50)
    print("  Rede Neural do Zero — Problema XOR")
    print("=" * 50)

    # Dados: cada coluna é um exemplo
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]], dtype=float)

    y = np.array([[0, 1, 1, 0]], dtype=float)

    print("\nDados de treino:")
    print("  x1  x2 → y")
    for i in range(4):
        print(f"   {int(X[0,i])}   {int(X[1,i])} → {int(y[0,i])}")

    print("\nTreinando...\n")
    params, historico = treinar(X, y, n_oculta=8, lr=0.5, epochs=5000)

    print("\n" + "=" * 50)
    print("Resultado final:")
    probs, classes = prever(X, params)
    print("\n  x1  x2 | Real | Previsto | Prob")
    print("  " + "-" * 38)
    for i in range(4):
        print(f"   {int(X[0,i])}   {int(X[1,i])} |  {int(y[0,i])}   |    {int(classes[0,i])}     | {probs[0,i]:.3f}")

    acuracia_final = np.mean(classes == y) * 100
    print(f"\nAcurácia final: {acuracia_final:.0f}%")
    print(f"Loss final:     {historico[-1]:.4f}")