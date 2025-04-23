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
        Value call(const std::vector<Value>& x);
        std::vector<Value*> parameters();
    

};

class Layer
{
    private:
        std::vector<Neuron> neurons;


    public:
        Layer(int n_in, int n_out);
        std::vector<Value> call(const std::vector<Value>& x);
        std::vector<Value*> parameters();
};


class MLP
{
    private:
        std::vector<Layer> layers;

    public:
        MLP(int n_in, const std::vector<int>& n_out);
        std::vector<Value> call(const std::vector<Value>& x);
        std::vector<Value*> parameters();
};




#endif