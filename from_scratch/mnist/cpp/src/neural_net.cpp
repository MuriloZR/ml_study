#include "neural_net.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
//#include <omp.h>

matrix carregar_arquivo(const std::string& caminho, int linhas, int colunas) {
    matrix mat;
    mat.linhas = linhas;
    mat.colunas = colunas;
    mat.m.resize(linhas * colunas);

    std::ifstream arquivo(caminho, std::ios::binary);
    if (!arquivo) {
        std::cerr << "Erro ao abrir " << caminho << "\n";
        exit(1);
    }

    arquivo.read(reinterpret_cast<char*>(mat.m.data()), mat.m.size() * sizeof(double));
    arquivo.close();

    return mat;
}

RedeNeural carregar_modelo(const std::vector<int>& topologia) {
    RedeNeural rede;
    rede.num_camadas = topologia.size() - 1;

    for (int i = 0; i < rede.num_camadas; ++i) {
        int linhas = topologia[i + 1];
        int colunas = topologia[i];

        std::string nome_w = "W" + std::to_string(i + 1) + ".bin";
        std::string nome_b = "b" + std::to_string(i + 1) + ".bin";

        rede.W.push_back(carregar_arquivo(nome_w, linhas, colunas));
        rede.b.push_back(carregar_arquivo(nome_b, linhas, 1));
    }
    return rede;
}

static void matrix_multiply(const matrix &A, const matrix &B, matrix &C) {
    C.linhas = A.linhas;
    C.colunas = B.colunas;
    C.m.resize(C.linhas * C.colunas, 0);

    //#pragma omp parallel for
    for (int i = 0; i < C.linhas; i++) {
        int i_offset {i * C.colunas};
        for (int k {0}; k < A.colunas; k++) {
            double temp {A.m[i_offset + k]};
            int k_offset {k * C.colunas};
            for (int j {0}; j < B.colunas; j++) {
                C.m[i_offset + j] += temp * B.m[k_offset + j];
            }
        }
    }
}

static void matrix_add_bias(matrix &A, const matrix &b) {
    for (int i = 0; i < A.linhas; i++) {
        int offset {i * A.linhas};
        for (int j = 0; j < A.colunas; j++) {
            A.m[offset + j] = b.m[i];
        }
    }
}

static matrix relu(const matrix &Z) {
    matrix mres{Z};
    for (auto &i : mres.m) i = std::max(static_cast<double>(0), i);
    return mres;
}

static void _softmax(matrix &Z) {
    double
        max_z {*std::max_element(Z.m.begin(), Z.m.end())},
        soma_exp {0.0};

    for (double &z : Z.m) {
        z = std::exp(z - max_z);
        soma_exp += z;
    }

    for (auto &z : Z.m) z/= soma_exp;
}

static matrix softmax(matrix Z) {
    _softmax(Z);
    return Z;
}

static int argmax(const matrix &A) {
    auto max_it = std::max_element(A.m.begin(), A.m.end());
    int classe_predominante = std::distance(A.m.begin(), max_it);
    return classe_predominante;
}

previsao prever(const matrix& X, const RedeNeural& rede) {
    matrix A {X};

    for (int i = 0; i < rede.num_camadas; ++i) {
        matrix Z;
        matrix_multiply(rede.W[i], A, Z);
        matrix_add_bias(Z, rede.b[i]);

        if (i == rede.num_camadas - 1) {
            A = softmax(Z);
        } else {
            A = relu(Z);
        }
    }

    previsao res;
    res.probs = A;
    res.classe = argmax(A);

    return res;
}