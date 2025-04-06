// value implementation file

#include "engine.hpp"
#include <iostream>


// overloaded functions
Value Value::operator+(const Value& other) const{
    Value* a = const_cast<Value*>(this);
    Value* b = const_cast<Value*>(&other);
    Value out(a->data + b->data, {a,b}, "+");
    out._backward = [a,b, &out](){
        a->grad += 1.0f * out.grad;
        b->grad += 1.0f * out.grad;
    };
    return out;
}

Value Value::operator-(const Value& other) const{
    Value* a = const_cast<Value*>(this);
    Value* b = const_cast<Value*>(&other);
    Value out(a->data - b->data, {a,b}, "-");
    out._backward = [a,b, &out](){
        a->grad += 1.0f * out.grad;
        b->grad -= 1.0f * out.grad;
    };
    return out;
}

Value Value::operator*(const Value& other) const{
    Value* a = const_cast<Value*>(this);
    Value* b = const_cast<Value*>(&other);
    Value out(a->data * b->data, {a,b}, "*");
    out._backward = [a,b, &out](){
        a->grad += b->data * out.grad;
        b->grad += a->data * out.grad;
    };
    return out;
}

Value Value::operator/(const Value& other) const{
    Value* a = const_cast<Value*>(this);
    Value* b = const_cast<Value*>(&other);
    if (b->data == 0.0f) {
        throw std::runtime_error("Division by zero");
    }
    Value out(a->data / b->data, {a,b}, "/");
    return out;
}

Value Value::operator^(const Value& other) const{
    Value* a = const_cast<Value*>(this);
    Value* b = const_cast<Value*>(&other);
    Value out(std::pow(a->data, b->data), {a,b}, "^");
    out._backward = [a,b, &out](){
        a->grad += b->data * std::pow(a->data, b->data - 1) * out.grad;
    };
    return out;
}


// tanh activation function
Value Value::tanh() const{
    float x = this->data;
    float t = (std::exp(2 * x) - 1) / (std::exp(2 * x) + 1);
    Value* a = const_cast<Value*>(this);
    Value out(t, {a}, "tanh");
    out._backward = [a, &out](){
        a->grad += (1 - out.data * out.data) * out.grad;
        
    };
    return out;
}

void Value::backward() const{
    // topological sort
    std::vector<Value*> topo;
    std::set<Value*> visited;
    
    std::function<void(Value*)> build_topo = [&](Value* v){
        if (visited.find(v) == visited.end()){
            visited.insert(v);
            for(Value* child : v->_prev){
                build_topo(child);
            }
            topo.push_back(v);
        }
    };
    build_topo(const_cast<Value*>(this));
    std::reverse(topo.begin(), topo.end());
    const_cast<Value*>(this)->grad = 1.0f;
    for(Value* node : topo){
        node->_backward();
    }
}


int main(){

    // inputs
    Value x1(2.0f, {}, "x1");
    Value x2(0.0f, {}, "x2");

    // weights
    Value w1(-3.0f, {}, "w1");
    Value w2(1.0f, {}, "w2");

    // bias
    Value b(6.7f, {}, "b");

    Value x1w1 = x1 * w1;
    Value x2w2 = x2 * w2;
    Value x1w1x2w2 = x1w1 + x2w2;
    Value n = x1w1x2w2 + b;

    // activation
    Value o = n.tanh();

    
    std::cout << "Forward pass results:" << std::endl;
    std::cout << "o: " << o.data << std::endl;


    // backward pass
    // o.grad = 1.0f;
    o.backward();
    std::cout << "Backward pass results:" << std::endl;
    std::cout << "Grad for x1: " << x1.grad << std::endl;
    std::cout << "Grad for x2: " << x2.grad << std::endl;
    std::cout << "Grad for w1: " << w1.grad << std::endl;
    std::cout << "Grad for w2: " << w2.grad << std::endl;
    std::cout << "Grad for b: " << b.grad << std::endl;
    

    return 0;
}

