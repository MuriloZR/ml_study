#pragma once

#include <string>
#include <vector>

// as matrizes da rede são na verdade grandes vetores,
// nos quais o acesso é feito com [i * n_colunas + j]
// fazer a implementação dessa forma é mais eficiente
// para o processador, por causa da cache
struct matrix {
    int linhas;
    int colunas;
    std::vector<double> m;
};

struct previsao {
    matrix probs; // y_pred
    int classe;
};

struct RedeNeural {
    std::vector<matrix> W;
    std::vector<matrix> b;
    int num_camadas;
};

RedeNeural carregar_modelo(const std::vector<int>& topologia);
matrix carregar_arquivo(const std::string& caminho, int linhas, int colunas);
previsao prever(const matrix& X, const RedeNeural& rede);