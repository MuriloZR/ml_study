# ============================================
# Rede Neural do Zero em R
# ============================================

# 1. FUNÇÕES DE ATIVAÇÃO (Preservando dimensões)
# ---------------------------------------------

relu <- function(z) {
  z[z < 0] <- 0  # Mantém o formato de matriz original
  return(z)
}

relu_grad <- function(z) {
  return((z > 0) * 1) # Retorna 1 onde for > 0, mantendo o formato
}

sigmoid <- function(z) {
  z <- pmin(pmax(z, -500), 500)
  return(1 / (1 + exp(-z)))
}

sigmoid_grad <- function(z) {
  s <- sigmoid(z)
  return(s * (1 - s))
}

# 2. FUNÇÃO DE PERDA
# ---------------------------------------------

binary_cross_entropy <- function(y_pred, y_true) {
  eps <- 1e-15
  y_pred <- pmin(pmax(y_pred, eps), 1 - eps)
  return(-mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)))
}

binary_cross_entropy_grad <- function(y_pred, y_true) {
  eps <- 1e-15
  y_pred <- pmin(pmax(y_pred, eps), 1 - eps)
  m <- ncol(y_true)
  return(-(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / m)
}

# 3. INICIALIZAÇÃO
# ---------------------------------------------

inicializar_pesos <- function(n_entrada, n_oculta, n_saida, seed = 42) {
  set.seed(seed)
  params <- list(
    "W1" = matrix(rnorm(n_oculta * n_entrada), nrow = n_oculta) * sqrt(2 / n_entrada),
    "b1" = rep(0, n_oculta), # Usando vetor para facilitar o sweep
    "W2" = matrix(rnorm(n_saida * n_oculta), nrow = n_saida) * sqrt(2 / n_oculta),
    "b2" = rep(0, n_saida)
  )
  return(params)
}

# 4. FORWARD PASS
# ---------------------------------------------

forward_pass <- function(X, params) {
  # Camada oculta
  Z1 <- sweep(params$W1 %*% X, 1, params$b1, "+")
  A1 <- relu(Z1)
  
  # Camada de saída
  Z2 <- sweep(params$W2 %*% A1, 1, params$b2, "+")
  A2 <- sigmoid(Z2)
  
  cache <- list(Z1 = Z1, A1 = A1, Z2 = Z2, A2 = A2, X = X)
  return(list(y_pred = A2, cache = cache))
}

# 5. BACKWARD PASS
# ---------------------------------------------

backward_pass <- function(y_true, params, cache) {
  m <- ncol(y_true)
  
  # Camada de saída
  dA2 <- binary_cross_entropy_grad(cache$A2, y_true)
  dZ2 <- dA2 * sigmoid_grad(cache$Z2)
  
  dW2 <- (dZ2 %*% t(cache$A1)) / m
  db2 <- rowMeans(dZ2)
  
  # Camada oculta
  dA1 <- t(params$W2) %*% dZ2
  dZ1 <- dA1 * relu_grad(cache$Z1)
  
  dW1 <- (dZ1 %*% t(cache$X)) / m
  db1 <- rowMeans(dZ1)
  
  return(list(dW1 = dW1, db1 = db1, dW2 = dW2, db2 = db2))
}

# 6. ATUALIZAÇÃO E TREINO
# ---------------------------------------------

atualizar_pesos <- function(params, grads, lr) {
  params$W1 <- params$W1 - lr * grads$dW1
  params$b1 <- params$b1 - lr * grads$db1
  params$W2 <- params$W2 - lr * grads$dW2
  params$b2 <- params$b2 - lr * grads$db2
  return(params)
}

treinar <- function(X, y, n_oculta = 8, lr = 0.5, epochs = 5000) {
  n_entrada <- nrow(X)
  params <- inicializar_pesos(n_entrada, n_oculta, 1)
  
  for (i in 1:epochs) {
    res <- forward_pass(X, params)
    grads <- backward_pass(y, params, res$cache)
    params <- atualizar_pesos(params, grads, lr)
    
    if (i %% 500 == 0) {
      loss <- binary_cross_entropy(res$y_pred, y)
      cat(sprintf("Época %d | Loss: %.4f\n", i, loss))
    }
  }
  return(params)
}

# 8. TESTE COM XOR
# ---------------------------------------------

# Dados (cada coluna é um exemplo)
X <- matrix(c(0,0, 0,1, 1,0, 1,1), nrow = 2)
y <- matrix(c(0, 1, 1, 0), nrow = 1)

params_finais <- treinar(X, y)

# Previsão final
final <- forward_pass(X, params_finais)
cat("\nProbabilidades Finais:\n")
print(round(final$y_pred, 3))