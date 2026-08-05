#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include <array>

// ==================================================================================
//
// Implementação do zero de uma rede neural simples para o problema XOR
//
// ==================================================================================
//
// TODO-list:
//  Generalizar as camadas da rede, ou seja, substituir os ["Wx"] ["bx"] por uma
//  implementação genérica

// as matrizes da rede são na verdade grandes vetores,
// nos quais o acesso é feito com [i * n_colunas + j]
// fazer a implementação dessa forma é mais eficiente
// para o processador

enum camadaID {
    camada_1 = 0,
    camada_2 = 1,
    num_camadas = 2
};

typedef struct matrix {
  std::vector<double> m;
  int n_linhas;
  int n_colunas;
} matrix;

// struct que armazena os valores dos pesos e bias de cada
// camada da rede
typedef struct camadas {
  std::array<matrix, num_camadas> W;
  std::array<std::vector<double>, num_camadas> b;
} camadas;

// struct que armazena os outputs de todas as camadas
// da rede para poder aplicar os gradientes
// usa smart pointers para evitar cópias desnecessárias
typedef struct output {
  matrix y_pred;
  std::unique_ptr<matrix> X;
  std::array<std::unique_ptr<matrix>, num_camadas> Z;
  std::array<std::unique_ptr<matrix>, num_camadas> A;
} output;

// struct que guarda os valores para ajustar
// os pesos e bias das camadas da rede
typedef struct gradientes {
  std::array<std::unique_ptr<matrix>, num_camadas> dW;
  std::array<std::unique_ptr<std::vector<double>>, num_camadas> db;
} gradientes;

// struct para facilitar pra printar o resultado do treino
typedef struct resultado_treino {
  camadas params;
  std::vector<double> loss_history;
} resultado_treino;

// struct que retorna a previsão da rede, usada para printar
typedef struct previsao {
  matrix y_pred;
  std::vector<int> classes;
} previsao;

double relu(double z) { return z > 0 ? z : 0; }

double relu_grad(double z) { return static_cast<double>(z > 0); }

double sigmoid(double z) {
  return 1.0 / (1 + std::exp(-std::clamp(z, -500.0, 500.0)));
}

double sigmoid_grad(double z) {
  double s = sigmoid(z);
  return s * (1 - s);
}

double binary_cross_entropy(const std::vector<double> &y_pred, const std::vector<double> &y_true) {
  size_t n = y_pred.size();
  if (n == 0)
    return 0.0;

  double eps = 1e-15;
  double loss = 0.0;

  for (size_t i = 0; i < n; ++i) {
    double pred_clipped = std::clamp(y_pred[i], eps, 1.0 - eps);

    double perda = y_true[i] * std::log(pred_clipped) +
                   (1.0 - y_true[i]) * std::log(1.0 - pred_clipped);

    loss += perda;
  }
  return -(loss / static_cast<double>(n));
}

matrix binary_cross_entropy_grad(const std::vector<double> &y_pred, const std::vector<double> &y_true) {
  size_t n = y_pred.size();
  matrix grads;
  grads.n_linhas = 1;
  grads.n_colunas = n;
  grads.m.resize(grads.n_linhas * grads.n_colunas);
  double eps = 1e-15;

  for (size_t i = 0; i < n; i++) {
    double p = std::clamp(y_pred[i], eps, 1.0 - eps);
    double y = y_true[i];

    grads.m[i] = -(y / p - (1.0 - y) / (1.0 - p)) / static_cast<double>(n);
  }

  return grads;
}

camadas inicializar_pesos(int n_entrada, int n_oculta, int n_saida,
                          int seed = 42) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> normal(0.0, 1.0);

  camadas params;

  double escala_W1 = std::sqrt(2.0 / n_entrada);
  double escala_W2 = std::sqrt(2.0 / n_oculta);

  params.W[camada_1].m.resize(n_oculta * n_entrada);
  params.W[camada_1].n_linhas = n_oculta;
  params.W[camada_1].n_colunas = n_entrada;
  params.b[camada_1].assign(n_oculta, 0.0);
  for (int i = 0; i < n_oculta; ++i) {
    for (int j = 0; j < n_entrada; ++j) {
      params.W[camada_1].m[i * params.W[camada_1].n_colunas + j] =
          normal(rng) * escala_W1;
    }
  }

  params.W[camada_2].m.resize(n_saida * n_oculta);
  params.W[camada_2].n_linhas = n_saida;
  params.W[camada_2].n_colunas = n_oculta;
  params.b[camada_2].assign(n_saida, 0.0);
  for (int i = 0; i < n_saida; ++i) {
    for (int j = 0; j < n_oculta; ++j) {
      params.W[camada_2].m[i * params.W[camada_2].n_colunas + j] =
          normal(rng) * escala_W2;
    }
  }

  return params;
}

// Função que multiplica matrizes e adiciona o bias respectivo
matrix mult_matrix_bias(const matrix& m1, const matrix& m2, const std::vector<double>& bias) {
    matrix mres;
    mres.n_linhas = m1.n_linhas;
    mres.n_colunas = m2.n_colunas;
    mres.m.resize(mres.n_linhas * mres.n_colunas);

    for (int i = 0; i < mres.n_linhas; i++) {
        for (int j = 0; j < mres.n_colunas; j++) {
            mres.m[i * mres.n_colunas + j] = bias[i];
        }
    }

    // Esses 'for' usam uma ideia chamada "loop i-k-j",
    // o propósito dessa ideia é evitar cache misses do processador
    // ela faz isso mudando a ordem dos loops para que os índices
    // aumentem sempre de 1 em 1 na memória
    for (int i = 0; i < m1.n_linhas; i++) {
        for (int k = 0; k < m1.n_colunas; k++) {
            double temp = m1.m[i * m1.n_colunas + k];
            for (int j = 0; j < m2.n_colunas; j++) {
                mres.m[i * mres.n_colunas + j] += temp * m2.m[k * m2.n_colunas + j];
            }
        }
    }

    return mres;
}


matrix relu_matrix(const matrix &m) {
  matrix mres;
  mres.m.resize(m.n_linhas * m.n_colunas);
  mres.n_linhas = m.n_linhas;
  mres.n_colunas = m.n_colunas;
  for (int i = 0; i < mres.n_linhas; i++) {
    for (int j = 0; j < mres.n_colunas; j++) {
      mres.m[i * mres.n_colunas + j] = relu(m.m[i * m.n_colunas + j]);
    }
  }
  return mres;
}

matrix sigmoid_matrix(const matrix &m) {
  matrix mres;
  mres.m.resize(m.n_linhas * m.n_colunas);
  mres.n_linhas = m.n_linhas;
  mres.n_colunas = m.n_colunas;
  for (int i = 0; i < mres.n_linhas; i++) {
    for (int j = 0; j < mres.n_colunas; j++) {
      mres.m[i * mres.n_colunas + j] = sigmoid(m.m[i * m.n_colunas + j]);
    }
  }
  return mres;
}

output forward_pass(const matrix &X, const camadas &params) {
  const matrix &W1 = params.W.at(camada_1);
  const matrix &W2 = params.W.at(camada_2);
  const std::vector<double> &b1 = params.b.at(camada_1);
  const std::vector<double> &b2 = params.b.at(camada_2);

  output cache;

  cache.X = std::make_unique<matrix>(X);
  cache.Z[camada_1] = std::make_unique<matrix>(mult_matrix_bias(W1, X, b1));
  cache.A[camada_1] = std::make_unique<matrix>(relu_matrix(*cache.Z[camada_1]));
  cache.Z[camada_2] = std::make_unique<matrix>(mult_matrix_bias(W2, *cache.A[camada_1], b2));
  cache.A[camada_2] = std::make_unique<matrix>(sigmoid_matrix(*cache.Z[camada_2]));
  cache.y_pred = *cache.A[camada_2];

  return cache;
}

matrix sigmoid_grad_vector(const matrix &v1, const matrix &v2) {
  matrix vres;
  vres.n_linhas = v1.n_linhas;
  vres.n_colunas = v1.n_colunas;
  vres.m.resize(vres.n_linhas * vres.n_colunas);
  for (int i = 0; i < v1.n_linhas; i++) {
    vres.m[i] = v1.m[i] * sigmoid_grad(v2.m[i]);
  }
  return vres;
}

matrix mult_matrix_div_m(matrix m1, matrix m2, double m) {
  matrix mres;
  mres.n_linhas = m1.n_linhas;
  mres.n_colunas = m2.n_colunas;
  mres.m.resize(mres.n_linhas * mres.n_colunas, 0);
  for (int i = 0; i < mres.n_linhas; i++) {
    for (int j = 0; j < mres.n_colunas; j++) {
      double soma{0.0};
      for (int k = 0; k < m1.n_colunas; k++) {
        soma += m1.m[i * m1.n_colunas + k] * m2.m[k * m2.n_colunas + j];
      }

      mres.m[i * mres.n_colunas + j] = soma / m;
    }
  }

  return mres;
}

matrix transposta(matrix m) {
  matrix mres;
  mres.n_linhas = m.n_colunas;
  mres.n_colunas = m.n_linhas;
  mres.m.resize(mres.n_linhas * mres.n_colunas);
  for (int i = 0; i < mres.n_linhas; i++) {
    for (int j = 0; j < mres.n_colunas; j++) {
      mres.m[i * mres.n_colunas + j] = m.m[j * m.n_colunas + i];
    }
  }
  return mres;
}

matrix produto_hadamard_sigmoid(matrix m1, matrix m2) {
  matrix mres;
  mres.n_linhas = m1.n_linhas;
  mres.n_colunas = m1.n_colunas;
  mres.m.resize(m1.m.size());
  for (int i = 0; i < mres.m.size(); i++) {
    mres.m[i] = m1.m[i] * sigmoid_grad(m2.m[i]);
  }

  return mres;
}

matrix produto_hadamard_relu(matrix m1, matrix m2) {
  matrix mres;
  mres.n_linhas = m1.n_linhas;
  mres.n_colunas = m1.n_colunas;
  mres.m.resize(m1.m.size());
  for (int i = 0; i < mres.m.size(); i++) {
    mres.m[i] = m1.m[i] * relu_grad(m2.m[i]);
  }

  return mres;
}

std::vector<double> mean(matrix m, int n) {
  std::vector<double> vres(m.n_linhas, 0.0);
  for (int i = 0; i < m.n_linhas; i++) {
    double soma = 0;
    for (int j = 0; j < m.n_colunas; j++) {
      soma += m.m[i * m.n_colunas + j];
    }
    vres[i] = soma / n;
  }

  return vres;
}

gradientes backward_pass(const std::vector<double> &y_true, camadas &params, output &cache) {
  int m = y_true.size();
  const matrix &W2 = params.W.at(camada_2);
  const matrix &Z1 = *cache.Z.at(camada_1);
  const matrix &A1 = *cache.A.at(camada_1);
  const matrix &Z2 = *cache.Z.at(camada_2);
  const matrix &A2 = *cache.A.at(camada_2);
  const matrix &X = *cache.X;

  gradientes grad;

  matrix dA2 = binary_cross_entropy_grad(A2.m, y_true);

  matrix dZ2 = produto_hadamard_sigmoid(dA2, Z2);

  grad.dW[camada_2] = std::make_unique<matrix>(mult_matrix_div_m(dZ2, transposta(A1), static_cast<double>(m)));
  grad.db[camada_2] = std::make_unique<std::vector<double>>(mean(dZ2, m));

  matrix dA1 = mult_matrix_div_m(transposta(W2), dZ2, 1);
  matrix dZ1 = produto_hadamard_relu(dA1, Z1);
  grad.dW[camada_1] = std::make_unique<matrix>(mult_matrix_div_m(dZ1, transposta(X), m));
  grad.db[camada_1] = std::make_unique<std::vector<double>>(mean(dZ1, m));

  return grad;
}

camadas atualizar_pesos(camadas params, gradientes &grad, double lr) {
  const matrix &dW1 = *grad.dW.at(camada_1);
  const std::vector<double> &db1 = *grad.db.at(camada_1);
  const matrix &dW2 = *grad.dW.at(camada_2);
  const std::vector<double> &db2 = *grad.db.at(camada_2);

  matrix &W1 = params.W.at(camada_1);
  std::vector<double> &b1 = params.b.at(camada_1);
  matrix &W2 = params.W.at(camada_2);
  std::vector<double> &b2 = params.b.at(camada_2);

  for (size_t i = 0; i < W1.m.size(); ++i) {
    W1.m.at(i) -= lr * dW1.m.at(i);
  }
  for (int i = 0; i < params.b[camada_1].size(); i++) {
    b1.at(i) -= lr * db1.at(i);
  }
  for (size_t i = 0; i < params.W[camada_2].m.size(); ++i) {
    W2.m.at(i) -= lr * dW2.m.at(i);
  }
  for (int i = 0; i < params.b[camada_2].size(); i++) {
    b2.at(i) -= lr * db2.at(i);
  }

  return params;
}

resultado_treino treinar(matrix X, matrix y, int n_oculta = 4, double lr = 0.1,
                         int epochs = 1000, bool verbose = true) {
  int n_entrada = X.n_linhas;
  int n_saida = 1;
  resultado_treino res;
  res.params = inicializar_pesos(n_entrada, n_oculta, n_saida);

  for (int i = 0; i < epochs; i++) {
    auto output = forward_pass(X, res.params);
    auto loss = binary_cross_entropy(output.y_pred.m, y.m);
    res.loss_history.push_back(loss);

    auto grads = backward_pass(y.m, res.params, output);

    res.params = atualizar_pesos(res.params, grads, lr);

    if (verbose && i % 500 == 0) {
      std::vector<int> y_classe(output.y_pred.m.size());
      for (int j = 0; j < output.y_pred.m.size(); j++) {
        y_classe[j] = static_cast<int>(output.y_pred.m[j] >= 0.5);
      }
      float acuracia{0.0};
      for (int j = 0; j < y_classe.size(); j++) {
        if (y_classe[j] == y.m[j])
          acuracia += 1;
      }
      acuracia /= static_cast<float>(y_classe.size());
      acuracia *= 100;
      printf("Epoca: %4d, Loss: %.4f, Acuracia: %.2f\n", i, loss, acuracia);
    }
  }

  return res;
}

previsao prever(matrix X, camadas params) {
  previsao prev;
  auto out = forward_pass(X, params);
  prev.y_pred.m.resize(out.y_pred.m.size());
  prev.classes.resize(out.y_pred.m.size());
  prev.y_pred = out.y_pred;
  for (int i = 0; i < prev.y_pred.m.size(); i++) {
    prev.classes[i] = static_cast<int>(prev.y_pred.m[i] >= 0.5);
  }
  return prev;
}

int main() {
  printf("Rede Neural do Zero, Problema XOR\n");
  matrix X;
  X.m = {0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0};
  X.n_linhas = 2;
  X.n_colunas = 4;
  matrix y{.m = {0, 1, 1, 0}, .n_linhas = 4, .n_colunas = 1};
  auto res = treinar(X, y, 8, 0.5, 5000);
  auto prev = prever(X, res.params);

  std::cout << "\n  x1  x2 | Real | Previsto | Prob" << std::endl;

  std::cout << "  " << std::string(38, '-') << std::endl;

  for (int i = 0; i < 4; i++) {
    int x1 = static_cast<int>(X.m[0 * X.n_colunas + i]);
    int x2 = static_cast<int>(X.m[1 * X.n_colunas + i]);
    int real = static_cast<int>(y.m[0 * y.n_colunas + i]);
    int previsto = static_cast<int>(prev.classes[i]);
    double prob = prev.y_pred.m[0 * prev.y_pred.n_colunas + i];

    std::cout << "    " << x1 << "   " << x2 << " |  " << real << "   |    "
              << previsto << "     | " << std::fixed << std::setprecision(3)
              << prob << std::endl;
  }
}
