/* Neuron, layer, mlp class*/

#include "mlp.hpp"
#include <iostream>
#include <random>

Neuron::Neuron(int n_in) : b(0.0f) {
    std::random_device rdev;
    std::mt19937 rgen(rdev());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    w.reserve(n_in);
    
    for (int i = 0; i < n_in; i++) {
        w.push_back(Value(dist(rgen)));
    }
    
    b = Value(dist(rgen));
}

Value Neuron::call(const std::vector<Value>& x) {
    if (x.size() != w.size()) {
        throw std::runtime_error("Input size doesn't match weight size");
    }
    
    Value act = b; 
    for (size_t i = 0; i < x.size(); i++) {
        act = act + (w[i] * x[i]);
    }

    return act.tanh();
}

std::vector<Value*> Neuron::parameters() {
    std::vector<Value*> params;
    
    params.reserve(w.size() + 1);
    for (auto& weight : w) {
        params.push_back(&weight);
    }
    
    params.push_back(&b);
    
    return params;
}

Layer::Layer(int n_in, int n_out) {
    neurons.reserve(n_out);
    for (int i = 0; i < n_out; i++) {
        neurons.emplace_back(n_in);
    }
}

std::vector<Value> Layer::call(const std::vector<Value>& x) {
    std::vector<Value> outputs;
    outputs.reserve(neurons.size());
    
    for (auto& neuron : neurons) {
        outputs.push_back(neuron.call(x));
    }
    
    return outputs;
}

std::vector<Value*> Layer::parameters() {
    std::vector<Value*> params;
    
    for (auto& neuron : neurons) {
        auto neuron_params = neuron.parameters();
        params.insert(params.end(), neuron_params.begin(), neuron_params.end());
    }
    
    return params;
}

MLP::MLP(int n_in, const std::vector<int>& n_out) {
    std::vector<int> sizes;
    sizes.push_back(n_in);
    sizes.insert(sizes.end(), n_out.begin(), n_out.end());
    
    layers.reserve(n_out.size());
    for (size_t i = 0; i < n_out.size(); i++) {
        layers.emplace_back(sizes[i], sizes[i+1]);
    }
}

std::vector<Value> MLP::call(const std::vector<Value>& x) {
    std::vector<Value> outputs = x;
    
    for (auto& layer : layers) {
        outputs = layer.call(outputs);
    }
    
    return outputs;
}

std::vector<Value*> MLP::parameters() {
    std::vector<Value*> params;
    
    for (auto& layer : layers) {
        auto layer_params = layer.parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    
    return params;
}




