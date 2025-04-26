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




