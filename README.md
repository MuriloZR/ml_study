# My Neural Network from Scratch (NumPy) 🧠

Este repositório contém implementações de Redes Neurais Artificiais construídas **totalmente do zero** utilizando apenas **NumPy** para operações matemáticas e álgebra linear. O objetivo deste projeto foi aprofundar o entendimento sobre o funcionamento interno do Backpropagation, funções de ativação e otimização de gradientes.

## 🚀 Destaques do Projeto
* **Zero Frameworks:** Sem dependências de Deep Learning (apenas NumPy para processamento e Sklearn para carga de dados).
* **Backpropagation Raiz:** Implementação manual das derivadas e da regra da cadeia.
* **Modularidade:** Código estruturado para suportar um número arbitrário de camadas e neurônios.

---

## ⚡ Desafio 1: O Problema XOR
Uma rede neural clássica para resolver um problema não linearmente separável.

* **Entrada:** 2 neurônios (Portas lógicas).
* **Objetivo:** Aprender a lógica XOR (0,0 -> 0; 0,1 -> 1; 1,0 -> 1; 1,1 -> 0).
* **Significado:** Demonstra a necessidade de camadas ocultas e funções de ativação não lineares para resolver problemas que uma simples regressão linear não consegue.

---

## 🔢 Desafio 2: MNIST (Dígitos Manuscritos)
O "Hello World" da visão computacional. A rede foi treinada para classificar dígitos de 0 a 9.

### Arquitetura Utilizada:
* **Input:** 784 neurônios (pixels 28x28).
* **Camadas Ocultas:** 128 e 64 neurônios com ativação **ReLU**.
* **Output:** 10 neurônios com ativação **Softmax**.
* **Inicialização:** He Initialization (otimizada para ReLU).
* **Loss:** Categorical Cross-Entropy.

### Resultados:
* **Acurácia obtida:** `97.80%` no conjunto de teste.
* **Épocas:** 30.
* **Otimização:** Mini-batch Gradient Descent.

---

## 🛠️ Tecnologias
* **Python 3.x**
* **NumPy** (Cálculo matricial)
* **Scikit-Learn** (Apenas para download do dataset e split de treino/teste)
* **Matplotlib** (Opcional, para plotar curvas de erro)

## 📂 Como executar
1. Instale as dependências:
   ```bash
   pip install numpy scikit-learn
   ```
2. Execute o script principal:
   ```bash
   python mnist_nn.py
   ```

---

## 🧠 O que eu aprendi:
* Como funciona o **Forward Pass** através de multiplicações de matrizes.
* A matemática por trás do **Backpropagation** e como os gradientes fluem da saída para a entrada.
* A importância da **Normalização** de dados para evitar a explosão de gradientes.
* A diferença entre **Batch**, **Stochastic** e **Mini-batch** Gradient Descent.