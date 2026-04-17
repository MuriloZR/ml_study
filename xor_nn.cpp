#include <iomanip>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <vector>

// ================================================================================== //
// Tentativa falha de implementar a rede neural para operação de XOR em C++
//
// Eu não recebo nada além de conhecimento por estar fazendo isso,
// e como já aprendi várias coisas novas, não vou debugar o código,
// mas o código dá esse erro:
//
// /usr/include/c++/16/bits/stl_vector.h:1272: constexpr std::vector<_Tp, _Alloc>::const_reference std::vector<_Tp, _Alloc>::operator[](size_type) const [with _Tp = double; _Alloc = std::allocator<double>; const_reference = const double&; size_type = long unsigned int]: Assertion '__n < this->size()' failed.
// Program received signal SIGABRT, Aborted.
// #0  0x00007ffff7a4d9ac in __pthread_kill_implementation () from /lib64/libc.so.6
// #1  0x00007ffff79f234e in raise () from /lib64/libc.so.6
// #2  0x00007ffff79d97b3 in abort () from /lib64/libc.so.6
// #3  0x00007ffff7c0b3c6 in std::__glibcxx_assert_fail(char const*, int, char const*, char const*) () from /lib64/libstdc++.so.6
// #4  0x0000000000405789 in std::vector<double, std::allocator<double> >::operator[] (this=0x7fffffffda50, __n=8) at /usr/include/c++/16/bits/stl_vector.h:1272
// #5  0x00000000004013cf in mult_matrix_bias (m1=..., m2=..., bias=std::vector of length 8, capacity 8 = {...}) at xor_nn.cpp:115
// #6  0x00000000004018a3 in forward_pass (X=..., params=...) at xor_nn.cpp:158
// #7  0x000000000040387b in treinar (X=..., y=..., n_oculta=8, lr=0.5, epochs=5000, verbose=true) at xor_nn.cpp:318
// #8  0x0000000000403f36 in main () at xor_nn.cpp:366
// ================================================================================== //

using vetor = std::vector<double>;

double relu(double z) {
    return z>0?z:0;
}

double relu_grad(double z) {
    return static_cast<double>(z > 0);
}

double sigmoid(double z) {
    return 1.0/(1 + std::exp(-std::clamp(z, -500.0, 500.0)));
}

double sigmoid_grad(double z) {
    double s = sigmoid(z);
    return s * (1 - s);
}

double binary_cross_entropy(const vetor& y_pred, const vetor& y_true) {
    size_t n = y_pred.size();
    if (n == 0) return 0.0;

    double eps = 1e-15;
    double loss = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double pred_clipped = std::clamp(y_pred[i], eps, 1.0 - eps);

        double perda = y_true[i] * std::log(pred_clipped) + (1.0 - y_true[i]) * std::log(1.0 - pred_clipped);

        loss += perda;
    }
    return -(loss / static_cast<double>(n));
}

typedef struct matriz {
    vetor m;
    int n_linhas;
    int n_colunas;
} matriz;

matriz binary_cross_entropy_grad(const vetor& y_pred, const vetor& y_true) {
    size_t n = y_pred.size();
    matriz grads;
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

typedef struct camadas {
    std::unordered_map<std::string, matriz> W;
    std::unordered_map<std::string, vetor> b;
} camadas;

camadas inicializar_pesos(int n_entrada, int n_oculta, int n_saida, int seed = 42) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    camadas params;

    double escala_W1 = std::sqrt(2.0 / n_entrada);
    double escala_W2 = std::sqrt(2.0 / n_oculta);

    params.W["W1"].m.resize(n_oculta * n_entrada);
    params.W["W1"].n_linhas = n_oculta;
    params.W["W1"].n_colunas = n_entrada;
    params.b["b1"].assign(n_oculta, 0.0);
    for (int i = 0; i < n_oculta; ++i) {
        for (int j = 0; j < n_entrada; ++j) {
            params.W["W1"].m[i * params.W["W1"].n_colunas + j] = normal(rng) * escala_W1;
        }
    }

    params.W["W2"].m.resize(n_saida * n_oculta);
    params.W["W1"].n_linhas = n_saida;
    params.W["W1"].n_colunas = n_oculta;
    params.b["b2"].assign(n_saida, 0.0);
    for (int i = 0; i < n_saida; ++i) {
        for (int j = 0; j < n_oculta; ++j) {
            params.W["W2"].m[i * params.W["W2"].n_colunas + j] = normal(rng) * escala_W2;
        }
    }

    return  params;
}

matriz mult_matrix_bias(const matriz& m1, const matriz& m2, const vetor& bias) {
    matriz mres;
    mres.m.resize(m1.n_linhas * m2.n_colunas, 0);
    mres.n_linhas = m1.n_linhas;
    mres.n_colunas = m2.n_colunas;
    for (int i = 0; i < mres.n_linhas; i++) {
        for (int j = 0; j < mres.n_colunas; j++) {
            double soma{0.0};
            for (int k = 0; k < m1.n_colunas; k++) {
                soma += m1.m[i * m1.n_colunas + k] * m2.m[k * m2.n_linhas + j];
            }
            mres.m[i * mres.n_colunas + j] = soma + bias[i];
        }
    }
    return mres;
}

matriz relu_matrix(const matriz& m) {
    matriz mres;
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

matriz sigmoid_matrix(const matriz& m) {
    matriz mres;
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

typedef struct output {
    matriz y_pred;
    std::unordered_map<std::string, matriz> cache;
} output;

output forward_pass(const matriz& X, const camadas& params) {
    matriz W1 = params.W.at("W1"), W2 = params.W.at("W2");
    vetor b1 = params.b.at("b1"), b2 = params.b.at("b2");

    auto Z1 = mult_matrix_bias(W1, X, b1);
    auto A1 = relu_matrix(Z1);
    auto Z2 = mult_matrix_bias(W2, A1, b2);
    auto A2 = sigmoid_matrix(Z2);

    output out;
    out.y_pred = A2;
    out.cache["Z1"] = Z1;
    out.cache["A1"] = A1;
    out.cache["Z2"] = Z2;
    out.cache["A2"] = A2;
    out.cache["X"] = X;

    return out;
}

matriz sigmoid_grad_vector(const matriz& v1, const matriz& v2) {
    matriz vres;
    vres.n_linhas = v1.n_linhas;
    vres.n_colunas = v1.n_colunas;
    vres.m.resize(vres.n_linhas * vres.n_colunas);
    for (int i = 0; i < v1.n_linhas; i++) {
        vres.m[i] = v1.m[i] * sigmoid_grad(v2.m[i]);
    }
    return vres;
}

matriz mult_matrix_div_m(matriz m1, matriz m2, double m) {
    matriz mres;
    mres.m.resize(m1.n_linhas * m2.n_colunas, 0);
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

matriz transposta(matriz m) {
    matriz mres;
    mres.n_linhas = m.n_colunas;
    mres.n_colunas = m.n_linhas;
    for (int i = 0; i < mres.n_linhas; i++) {
        for (int j = 0; j < mres.n_colunas; j++) {
            mres.m[i * mres.n_colunas + j] = m.m[j * m.n_colunas + i];
        }
    }
    return mres;
}

matriz produto_hadamard_sigmoid(matriz m1, matriz m2) {
    matriz mres;
    mres.n_linhas = m1.n_linhas;
    mres.n_colunas = m1.n_colunas;
    mres.m.resize(m1.m.size());
    for (int i = 0; i < mres.m.size(); i++) {
        mres.m[i] = m1.m[i] * sigmoid_grad(m2.m[i]);
    }

    return mres;
}

matriz produto_hadamard_relu(matriz m1, matriz m2) {
    matriz mres;
    mres.n_linhas = m1.n_linhas;
    mres.n_colunas = m1.n_colunas;
    mres.m.resize(m1.m.size());
    for (int i = 0; i < mres.m.size(); i++) {
        mres.m[i] = m1.m[i] * relu_grad(m2.m[i]);
    }

    return mres;
}

vetor mean(matriz m, int n) {
    vetor vres(m.n_linhas, 0.0);
    for (int i = 0; i < m.n_linhas; i++) {
        double soma = 0;
        for (int j = 0; j < m.n_colunas; j++) {
            soma += m.m[i * m.n_colunas + j];
        }
        vres[i] = soma / n;
    }

    return vres;
}

typedef struct gradientes {
    std::unordered_map<std::string, matriz> dW;
    std::unordered_map<std::string, vetor> db;
} gradientes;

gradientes backward_pass(const vetor& y_true, camadas params, output cache) {
    int m = y_true.size();
    matriz W2 = params.W["W2"];
    matriz Z1 = cache.cache["Z1"];
    matriz A1 = cache.cache["A1"];
    matriz Z2 = cache.cache["Z2"];
    matriz A2 = cache.cache["A2"];
    matriz X = cache.cache["X"];

    matriz dA2 = binary_cross_entropy_grad(A2.m, y_true);

    // dZ2 = dA2 * sigmoid_grad(Z2)
    matriz dZ2 = produto_hadamard_sigmoid(dA2, Z2);

    matriz dW2 = mult_matrix_div_m(dZ2, transposta(A1), static_cast<double>(m));

    // db2 = sum(dZ2, axis=1) / m
    vetor db2 = mean(dZ2, m);

    matriz dA1 = mult_matrix_div_m(transposta(W2), dZ2, 1);
    matriz dZ1 = produto_hadamard_relu(dA1, Z1);
    matriz dW1 = mult_matrix_div_m(dZ1, transposta(X), m);
    vetor db1 = mean(dZ1, m);

    gradientes grad;
    grad.dW["W1"] = dW1;
    grad.dW["W2"] = dW2;
    grad.db["db1"] = db1;
    grad.db["db2"] = db2;

    return grad;
}

camadas atualizar_pesos(camadas params, gradientes grad, double lr) {
    for (size_t i = 0; i < params.W["W1"].m.size(); ++i) {
        params.W["W1"].m[i] -= lr * grad.dW.at("W1").m[i];
    }
    for (int i = 0; i < params.b["b1"].size(); i++) {
        params.b["b1"][i] -= lr * grad.db["db1"][i];
    }
    for (size_t i = 0; i < params.W["W2"].m.size(); ++i) {
        params.W["W2"].m[i] -= lr * grad.dW.at("W2").m[i];
    }
    for (int i = 0; i < params.b["b2"].size(); i++) {
        params.b["b2"][i] -= lr * grad.db["db2"][i];
    }

    return params;
}

typedef struct resultado_treino {
    camadas params;
    vetor loss_history;
} resultado_treino;

resultado_treino treinar(matriz X, matriz y, int n_oculta=4, double lr=0.1, int epochs=1000, bool verbose=true) {
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
                y_classe[i] = static_cast<int>(output.y_pred.m[i] >= 0.5);
            }
            float acuracia{0.0};
            for (int j = 0; j < y_classe.size(); j++) {
                if (y_classe[i] == y.m[i]) acuracia += 1;
            }
            acuracia /= static_cast<float>(y_classe.size());
            acuracia *= 100;
            printf("Epoca: %4d, Loss: %.4f, Acuracia: %.2f", i, loss, acuracia);
        }
    }

    return res;
}

typedef struct previsao {
    matriz y_pred;
    std::vector<int> classes;
} previsao;

previsao prever(matriz X, camadas params) {
    previsao prev;
    auto out = forward_pass(X, params);
    prev.y_pred = out.y_pred;
    for (int i = 0; i < prev.y_pred.m.size(); i++) {
        prev.classes[i] = static_cast<int>(prev.y_pred.m[i] >= 0.5);
    }
    return prev;
}

int main() {
    printf("Rede Neural do Zero, Problema XOR\n");
    matriz X;
    X.m = {0.0,0.0,1.0,1.0,0.0,1.0,0.0,1.0};
    X.n_linhas = 4;
    X.n_colunas = 2;
    matriz y{.m = {0,1,1,0}, .n_linhas = 4, .n_colunas = 1};
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

        std::cout << "    " << x1 << "   " << x2 << " |  "
                  << real << "   |    " << previsto << "     | "
                  << std::fixed << std::setprecision(3) << prob << std::endl;
    }
}