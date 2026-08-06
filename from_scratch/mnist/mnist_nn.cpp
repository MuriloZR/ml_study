#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include <array>
#include <fstream>

// ==================================================================================
//
// Implementação de uma rede neural simples para o Dataset do MNIST.
// O treinamento inteiro é feito com python e essa versão em C++ apenas
// importa os pesos da rede treinada e faz previsões
//
// ==================================================================================

enum camadaID {
    camada1 = 0,
    camada2 = 1,
    camada3 = 2,
    num_camadas = 3
};

typedef struct matrix {
  std::vector<double> m;
  int n_linhas;
  int n_colunas;
} matrix;

typedef struct camadas {
  std::array<matrix, num_camadas> W;
  std::array<std::vector<double>, num_camadas> b;
} camadas;

typedef struct previsao {
  matrix y_pred;
  int classe;
} previsao;

// Função para ler o arquivo binário direto para a matriz
matrix carregar_pesos(const std::string& caminho, int linhas, int colunas) {
    matrix mat;
    mat.n_linhas = linhas;
    mat.n_colunas = colunas;
    mat.m.resize(linhas * colunas);

    std::ifstream arquivo(caminho, std::ios::binary);
    if (!arquivo) {
        std::cerr << "Erro ao abrir " << caminho << "\n";
        exit(1);
    }

    // Lê os bytes do disco e joga direto no vetor da matriz
    arquivo.read(reinterpret_cast<char*>(mat.m.data()), mat.m.size() * sizeof(double));
    arquivo.close();

    return mat;
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

double relu(double z) { return z > 0 ? z : 0; }

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

void softmax(matrix &m) {
    double
        max_z {*std::max_element(m.m.begin(), m.m.end())},
        soma_exp {0.0};

    for (double &z : m.m) {
        z = std::exp(z - max_z);
        soma_exp += z;
    }

    for (auto &z : m.m) z/= soma_exp;
}

matrix softmax_matrix(matrix m) {
    softmax(m);
    return m;
}

matrix forward_pass(const matrix &X, const camadas &params) {
    const matrix
        &W1 = params.W.at(camada1),
        &W2 = params.W.at(camada2),
        &W3 = params.W.at(camada3);

    const std::vector<double>
        &b1 = params.b.at(camada1),
        &b2 = params.b.at(camada2),
        &b3 = params.b.at(camada3);

    std::array<matrix, num_camadas> pipeline;

    pipeline[camada1] = relu_matrix(mult_matrix_bias(W1, X, b1));
    pipeline[camada2] = relu_matrix(mult_matrix_bias(W2, pipeline[camada1], b2));
    pipeline[camada3] = softmax_matrix(mult_matrix_bias(W3, pipeline[camada2], b3));

    return pipeline[camada3];
}

previsao prever(const matrix& X, const camadas& params) {
    previsao prev;
    prev.y_pred = forward_pass(X, params);
    auto max_it = std::max_element(prev.y_pred.m.begin(), prev.y_pred.m.end());
    int classe_predominante = std::distance(prev.y_pred.m.begin(), max_it);
    prev.classe = {classe_predominante};
    return prev;
}

int main() {
    camadas params;
    int
        entrada {784},
        oculta1 {128},
        oculta2 {64},
        saida {10};

    matrix X_teste = carregar_pesos("imagem_teste.bin", entrada, 1);
    params.W[camada1] = carregar_pesos("W1.bin", oculta1, entrada);
    params.b[camada1] = carregar_pesos("b1.bin", oculta1, 1).m;
    params.W[camada2] = carregar_pesos("W2.bin", oculta2, oculta1);
    params.b[camada2] = carregar_pesos("b2.bin", oculta2, 1).m;
    params.W[camada3] = carregar_pesos("W3.bin", saida, oculta2);
    params.b[camada3] = carregar_pesos("b3.bin", saida, 1).m;

    previsao res = prever(X_teste, params);

    std::cout << "\n--- Probabilidades no C++ ---\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << "Classe " << i << ": "
                  << std::fixed << std::setprecision(6)
                  << res.y_pred.m[i] << "\n";
    }

    std::cout << "Classe prevista (Argmax): " << res.classe << "\n";
}