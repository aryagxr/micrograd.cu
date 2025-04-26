#include "mlp.hpp"
#include <iostream>
#include <vector>

int main() {
    // Create a single neuron with 3 inputs
    Neuron neuron(3);
    
    // Create input
    std::vector<Value> x = {Value(2.0f), Value(3.0f), Value(-1.0f)};
    
    // Forward pass
    std::cout << "Running neuron forward pass..." << std::endl;
    Value output = neuron.call(x);
    std::cout << "Neuron output: " << output.data << std::endl;
    
    // Backward pass
    std::cout << "Running backward pass..." << std::endl;
    output.backward();
    
    // Check parameter gradients
    auto params = neuron.parameters();
    std::cout << "Neuron has " << params.size() << " parameters" << std::endl;
    std::cout << "Parameter gradients:" << std::endl;
    for (size_t i = 0; i < params.size(); i++) {
        std::cout << "  param[" << i << "] = " << params[i]->data 
                  << ", grad = " << params[i]->grad << std::endl;
    }
    
    return 0;
}