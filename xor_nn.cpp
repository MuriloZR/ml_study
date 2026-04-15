#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <vector>

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

double binary_cross_entropy(const std::vector<double>& y_pred, const std::vector<double>& y_true) {
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

std::vector<double> binary_cross_entropy_grad(const std::vector<double>& y_pred, const std::vector<double>& y_true) {
    size_t n = y_pred.size();
    std::vector<double> grads(n);
    double eps = 1e-15;

    for (size_t i = 0; i < n; i++) {
        double p = std::clamp(y_pred[i], eps, 1.0 - eps);
        double y = y_true[i];

        grads[i] = -(y / p - (1.0 - y) / (1.0 - p)) / static_cast<double>(n);
    }

    return grads;
}

typedef struct camadas {
    std::unordered_map<std::string, std::vector<std::vector<double>>> W;
    std::unordered_map<std::string, std::vector<double>> b;
} camadas;

camadas inicializar_pesos(int n_entrada, int n_oculta, int n_saida, int seed = 42) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    camadas params;

    double escala_W1 = std::sqrt(2.0 / n_entrada);
    double escala_W2 = std::sqrt(2.0 / n_oculta);

    params.W["W1"].resize(n_oculta, std::vector<double>(n_entrada));
    params.b["b1"].assign(n_oculta, 0.0);
    for (int i = 0; i < n_oculta; ++i) {
        for (int j = 0; j < n_entrada; ++j) {
            params.W["W1"][i][j] = normal(rng) * escala_W1;
        }
    }

    params.W["W2"].resize(n_saida, std::vector<double>(n_oculta));
    params.b["b2"].assign(n_saida, 0.0);
    for (int i = 0; i < n_saida; ++i) {
        for (int j = 0; j < n_oculta; ++j) {
            params.W["W2"][i][j] = normal(rng) * escala_W2;
        }
    }

    return  params;
}

std::vector<std::vector<double>> mult_matrix_bias(const std::vector<std::vector<double>> m1, const std::vector<std::vector<double>> m2, const std::vector<double>& bias) {
    std::vector<std::vector<double>> mres(m1.size(), std::vector<double>(m2[0].size(), 0));
    for (int i = 0; i < m1.size(); i++) {
        for (int j = 0; j < m2[0].size(); j++) {
            double soma = 0.0;
            for (int k = 0; k < m1[0].size(); k++) {
                soma += m1[i][k] * m2[k][j];
            }
            mres[i][j] = soma + bias[i];
        }
    }
    return mres;
}

std::vector<std::vector<double>> relu_matrix(const std::vector<std::vector<double>>& m) {
    std::vector<std::vector<double>> mres(m.size(), std::vector<double>(m[0].size()));
    for (int i = 0; i < mres.size(); i++) {
        for (int j = 0; j < mres[0].size(); j++) {
            mres[i][j] = relu(m[i][j]);
        }
    }
    return mres;
}

std::vector<std::vector<double>> sigmoid_matrix(const std::vector<std::vector<double>>& m) {
    std::vector<std::vector<double>> mres(m.size(), std::vector<double>(m[0].size()));
    for (int i = 0; i < mres.size(); i++) {
        for (int j = 0; j < mres[0].size(); j++) {
            mres[i][j] = sigmoid(m[i][j]);
        }
    }
    return mres;
}

typedef struct output {
    std::vector<std::vector<double>> y_pred;
    std::unordered_map<std::string, std::vector<std::vector<double>>> cache;
} output;

output forward_pass(const std::vector<std::vector<double>>& X, const camadas& params) {
    std::vector<std::vector<double>> W1 = params.W.at("W1"), W2 = params.W.at("W2");
    std::vector<double> b1 = params.b.at("b1"), b2 = params.b.at("b2");

    auto Z1 = mult_matrix_bias(W1, X, b1);
    auto A1 = relu_matrix(Z1);
    auto Z2 = mult_matrix_bias(W2, A1, b2);
    auto A2 = sigmoid_matrix(Z2);

    output out;
    out.y_pred = A2;
    out.cache["Z1"] = Z1;
    out.cache["A1"] = A1;
    out.cache["Z2"] = Z2;
    out.cache["A1"] = A2;
    out.cache["X"] = X;

    return out;
}

int main() {
    std::cout << relu_grad(0.0) << std::endl;
}