// neural net class definitions - neuron, layer, mlp
#ifndef MLP_HPP
#define MLP_HPP

#include "engine.hpp"

class Neuron
{
    private:
    std::vector<Value> w;
    Value b;

    public:
        
        Neuron(int n_in);
        Value forward(const std::vector<Value>& x);
        std::vector<Value*> parameters();
    

};



#endif