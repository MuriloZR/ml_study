import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

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
    return (y_pred - y_true)

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

#calcula a saída da rede, com um número generalizado de camadas
def forward_pass(X, params):
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

def backward_pass(y_true, params: dict, cache: dict):
    m = y_true.shape[1]
    W3 = params["W3"]
    W2 = params["W2"]
    Z1, A1, Z2, A2, Z3, A3, X = cache["Z1"], cache["A1"], cache["Z2"], cache["A2"], cache["Z3"], cache["A3"], cache["X"]

    # ── Camada de saída (camada 3) ──
    # Gradiente da loss em relação à saída
    dZ3 = cce_softmax_grad(A3, y_true) / m
    
    # Gradientes dos pesos da camada 3
    dW3 = dZ3 @ A2.T
    db3 = np.sum(dZ3, axis=1, keepdims=True)

    # ── Camada oculta (camada 2) ──
    # Propaga o gradiente para trás pela camada 3
    dA2 = W3.T @ dZ3

    # Gradiente de A2 em relação a Z2 (derivada do ReLU)
    dZ2 = dA2 * relu_grad(Z2)

    # Gradientes dos pesos da camada 2
    dW2 = dZ2 @ A1.T
    db2 = np.sum(dZ2, axis=1, keepdims=True)

    # ── Camada oculta (camada 1) ──
    # Propaga o gradiente para trás pela camada 2
    dA1 = W2.T @ dZ2

    # Gradiente de A1 em relação a Z1 (derivada do ReLU)
    dZ1 = dA1 * relu_grad(Z1)

    # Gradientes dos pesos da camada 1
    dW1 = dZ1 @ X.T
    db1 = np.sum(dZ1, axis=1, keepdims=True)

    gradientes = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}
    return gradientes

#TODO
def backward_pass_loop(y_true, params, cache):
    m = y_true.shape[1]
    W, Z, A = [], [], []
    dW, db, dZ, dA = [], [], [], []
    i = 1
    while f"W{i}" in params:
        W.append(params[f"W{i}"])
        i+=1
    
    i = 1
    while f"Z{i}" in cache:
        Z.append(cache[f"Z{i}"])
        i+=1

    i = 1
    while f"A{i+1}" in cache:
        A.append(cache[f"A{i}"])
        i+=1

    X = cache["X"]

    db.append

def atualizar_pesos(params: dict, gradientes: dict, lr):
    i = 1
    while f"W{i}" in params:
        params[f"W{i}"] -= lr * gradientes[f"dW{i}"]
        params[f"b{i}"] -= lr * gradientes[f"db{i}"]
        i+=1
    return params

def treinar(X, y_true, camadas, lr, epochs=1000, verbose=False):
    params = inicializar_pesos(camadas)
    historico_loss = []

    for epoch in range(epochs):
        y_pred, cache = forward_pass(X, params)

        loss = categorical_cross_entropy(y_pred, y_true)
        historico_loss.append(loss)

        gradientes = backward_pass(y_true, params, cache)

        params = atualizar_pesos(params, gradientes, lr)

        if verbose and epoch % 100 == 0:
            previsoes = np.argmax(y_pred, axis=0)
            reais = np.argmax(y_true, axis=0)
            acuracia = np.mean(previsoes == reais) * 100
            print(f"Época {epoch} | Loss: {loss:.4f} | Acurácia: {acuracia:.2f}%")

    return params, historico_loss

def prever(X, params):
    probabilidades, _ = forward_pass(X, params)
    classes = np.argmax(probabilidades, axis=0)
    return probabilidades, classes

if __name__ == "__main__":
    # --- 1. CARREGAMENTO E PRÉ-PROCESSAMENTO ---
    print("Carregando MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')

    X = mnist["data"].astype('float32') / 255.0  # Normalização
    y = mnist["target"].astype(int)

    # Separando 80% para treino e 20% para teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Transpondo para o formato que sua rede espera: (Features, Amostras)
    X_train = X_train.T
    X_test = X_test.T

    # One-hot encoding manual para o treino
    def to_one_hot(y, k=10):
        one_hot = np.zeros((k, y.size))
        one_hot[y, np.arange(y.size)] = 1
        return one_hot

    y_train_oh = to_one_hot(y_train)

    # --- 2. CONFIGURAÇÃO DA REDE ---
    # Entrada: 784 | Ocultas: 128, 64 | Saída: 10
    camadas = [784, 128, 64, 10]
    lr = 0.1
    epochs = 30  # Com os dados corrigidos, 30 épocas já dão um resultado incrível
    # --- 3. LOOP DE TREINAMENTO (COM MINI-BATCHES) ---
    print("Iniciando treinamento...")
    params = inicializar_pesos(camadas)
    batch_size = 64
    m_train = X_train.shape[1]
    
    for epoch in range(epochs):
        # Shuffle (embaralhar) os dados a cada época
        perm = np.random.permutation(m_train)
        X_train_shuffled = X_train[:, perm]
        y_train_shuffled = y_train_oh[:, perm]
        
        for i in range(0, m_train, batch_size):
            # Seleciona o lote
            x_batch = X_train_shuffled[:, i:i+batch_size]
            y_batch = y_train_shuffled[:, i:i+batch_size]
            
            # Forward, Backward e Update
            y_pred, cache = forward_pass(x_batch, params)
            grads = backward_pass(y_batch, params, cache)
            params = atualizar_pesos(params, grads, lr)
        
        # --- 4. AVALIAÇÃO PARCIAL ---
        # A cada época, testamos no set de validação (X_test)
        prob, classes_preditas = prever(X_test, params)
        acuracia = np.mean(classes_preditas == y_test) * 100
        print(f"Época {epoch+1}/{epochs} | Acurácia no Teste: {acuracia:.2f}%")
    
    print("Treinamento concluído!")