#include "mlp.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

int main() {
    std::cout << "Creating dataset..." << std::endl;
    // Create dataset similar to Python example
    std::vector<std::vector<Value>> xs = {
        {Value(2.0f), Value(3.0f), Value(-1.0f)},
        {Value(3.0f), Value(2.0f), Value(-1.0f)},
        {Value(4.0f), Value(5.0f), Value(-1.0f)},
        {Value(5.0f), Value(4.0f), Value(-1.0f)}
    };
    
    std::vector<Value> ys = {
        Value(1.0f), Value(-1.0f), Value(-1.0f), Value(1.0f)
    };

    std::cout << "Creating MLP..." << std::endl;
    // Create MLP model
    std::vector<int> layer_sizes = {4, 4, 1};
    MLP mlp(3, layer_sizes);
    
    std::cout << "Starting training loop..." << std::endl;
    // Training loop
    for (int k = 0; k < 20; k++) {
        std::cout << "Iteration " << k << " forward pass..." << std::endl;
        // Forward pass
        std::vector<Value> y_pred;
        for (const auto& x : xs) {
            auto output = mlp.call(x);
            y_pred.push_back(output[0]);
        }
        
        std::cout << "Calculating loss..." << std::endl;
        // Calculate loss (MSE)
        Value loss(0.0f);
        for (size_t i = 0; i < ys.size(); i++) {
            Value diff = y_pred[i] - ys[i];
            loss = loss + (diff * diff);
        }
        
        std::cout << "Getting parameters..." << std::endl;
        auto params = mlp.parameters();
        std::cout << "Got " << params.size() << " parameters" << std::endl;
        
        std::cout << "Zeroing gradients..." << std::endl;
        // Backward pass - first zero all gradients
        for (auto* p : params) {
            if (p == nullptr) {
                std::cout << "ERROR: Null parameter pointer!" << std::endl;
                return 1;
            }
            p->grad = 0.0f;
        }
        
        std::cout << "Running backward pass..." << std::endl;
        // Perform backpropagation
        loss.backward();
        
        std::cout << "Updating weights..." << std::endl;
        // Update weights (gradient descent)
        float learning_rate = 0.001f;
        for (auto* p : params) {
            p->data -= learning_rate * p->grad;
        }
        
        std::cout << "Iteration " << k << ", Loss: " << loss.data << std::endl;
    }
    
    return 0;
}