#ifndef ENGINE_HPP
#define ENGINE_HPP

#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <cmath>
#include <set>
#include <memory>
#include <algorithm>

// value class declaration
class Value
{
    public:
    float data;
    float grad;
    std::string _op;
    std::vector<Value*> _prev;
    std::function<void()> _backward;

    /* might have to make data members private
       and call through getters & setters. */

    
    Value(float data, const std::vector<Value*>& _prev = {}, const std::string& _op = ""){
        this->data = data;
        this->grad = 0.0f;
        this->_prev = _prev;
        this->_op = _op;
        this->_backward = [](){};
    }

    // overloading operators
    Value operator+(const Value&) const;
    Value operator-(const Value&) const;
    Value operator*(const Value&) const;
    Value operator/(const Value&) const;
    Value operator^(const Value&) const;


    Value tanh() const;

    void backward() const;

   

};

#endif