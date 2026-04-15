#include <iostream>
#include <cmath>
#include <algorithm>

float relu(float z) {
    return z>0?z:0;
}

float relu_grad(float z) {
    return static_cast<float>(z > 0);
}

int main() {
    std::cout << relu_grad(0.0) << std::endl;
}