// value implementation file

#include "engine.hpp"
#include <iostream>


// overloaded functions
Value Value::operator+(const Value& other) const{
    Value out(this->data + other.data, {const_cast<Value*>(this), const_cast<Value*>(&other)}, "+");
    return out;
}

Value Value::operator-(const Value& other) const{
    Value out(this->data - other.data, {const_cast<Value*>(this), const_cast<Value*>(&other)}, "-");
    return out;
}

Value Value::operator*(const Value& other) const{
    Value out(this->data * other.data, {const_cast<Value*>(this), const_cast<Value*>(&other)}, "*");
    return out;
}

Value Value::operator/(const Value& other) const{
    if (other.data == 0.0f) {
        throw std::runtime_error("Division by zero");
    }
    Value out(this->data / other.data, {const_cast<Value*>(this), const_cast<Value*>(&other)}, "/");
    return out;
}

Value Value::operator^(const Value& other) const{
    Value out(std::pow(this->data, other.data), {const_cast<Value*>(this), const_cast<Value*>(&other)}, "^");
    return out;
}


// tanh activation function
Value Value::tanh() const{
    float x = this->data;
    float t = (std::exp(2 * x) - 1) / (std::exp(2 * x) + 1);
    Value out(t, {const_cast<Value*>(this)}, "tanh");
    

    return out;
}


int main(){
    Value a(6.0f);
    Value b(3.0f);

    std::vector<Value*> prev = {&a, &b};
    Value c = a + b;
    Value d = a - b;
    Value e = a * b;
    Value f = a / b;
    Value g = a ^ b;

    // Access and print the data, _prev, and _op of v3
    std::cout << "Value data: " << c.data << std::endl;
    std::cout << "Value operation: " << c._op << std::endl;


    std::cout << "Value predecessors: ";
    for (const auto& p : c._prev) {
        std::cout << p->data << " ";
    }
    std::cout << std::endl;

    std::cout << "Value data: " << d.data << std::endl;
    std::cout << "Value operation: " << d._op << std::endl;
    std::cout << "Value predecessors: ";
    for (const auto& p : d._prev) {
        std::cout << p->data << " ";
    }

    std::cout << std::endl;

    std::cout << "Value data: " << e.data << std::endl;
    std::cout << "Value operation: " << e._op << std::endl;
    std::cout << "Value predecessors: ";
    for (const auto& p : e._prev) {
        std::cout << p->data << " ";
    }
    std::cout << std::endl;

    std::cout << "Value data: " << f.data << std::endl;
    std::cout << "Value operation: " << f._op << std::endl;
    std::cout << "Value predecessors: ";
    for (const auto& p : f._prev) {
        std::cout << p->data << " ";
    }
    std::cout << std::endl;

    std::cout << "Value data: " << g.data << std::endl;
    std::cout << "Value operation: " << g._op << std::endl;
    std::cout << "Value predecessors: ";
    for (const auto& p : g._prev) {
        std::cout << p->data << " ";
    }
    std::cout << std::endl;

    return 0;
}

