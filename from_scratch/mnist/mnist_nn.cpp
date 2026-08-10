#include <iomanip>
#include <iostream>
#include <vector>
#include "../include/neural_net.hpp"

// ==================================================================================
//
// Implementação de uma rede neural simples para o Dataset do MNIST.
// O treinamento inteiro é feito com python e essa versão em C++ apenas
// importa os pesos da rede treinada e faz previsões
//
// ==================================================================================

int main() {
    std::vector<int> topologia = {784, 128, 64, 10};

    RedeNeural modelo = carregar_modelo(topologia);
    matrix X_teste = carregar_arquivo("imagem_teste.bin", 784, 1);

    previsao res = prever(X_teste, modelo);

    std::cout << "\n--- Probabilidades ---\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << "Classe " << i << ": "
                  << std::fixed << std::setprecision(6)
                  << res.probs.m[i] << "\n";
    }

    std::cout << "Classe prevista (Argmax): " << res.classe << "\n";
}