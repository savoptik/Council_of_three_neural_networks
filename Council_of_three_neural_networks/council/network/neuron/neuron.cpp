//
//  neuron.cpp
//  forward_direction_neural_network
//
//  Created by Admin on 11/10/2018.
//  Copyright © 2018 Артём Семёнов. All rights reserved.
//

#include "neuron.hpp"
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>
using namespace std;

double& neuron::getOutputValue() {
    return outputValue; // возвращаем выходное значение
}

double& neuron::getLocalGradient() {
    return localGradient; // возвращаем локальный градиент
}

int neuron::getActivationFunction() { 
    return activationFunction; // возвращаем функцию активации
}

std::vector<double> &neuron::getVectorOfWeights() { 
    return vectorOfWeights; // возвращаем ссылку на вектор весов
}

neuron::neuron(int size, const int acFunc, const double afa, const double afb) {
    std::default_random_engine dra(static_cast<unsigned int>(time(0)));
//    std::mt19937 gen(rd.entropy());
    std::uniform_real_distribution<> urd(-0.8, 0.8);
    activationFunction = acFunc; // записываем функцию активации
    // резервируем память под векторы
    inSignal.resize(size);
    vectorOfWeights.resize(size);
    // заполняем вектор весов случайными числами
    for (int i = 0; i < vectorOfWeights.size(); i++) {
//        vectorOfWeights[i] = double(rand() % 16000)/10000 - 1.6;
//        system("sleep 1");
        double t = urd(dra);
        vectorOfWeights[i] = t;
    }
    // сохраняем параметры функции активации
    a = afa;
    b = afb;
}

std::vector<double> &neuron::getInputVector() { 
    return inSignal; // возвращаем ссылку на вектор входных сигналов
}

void neuron::theCalculationOfTheOutputValue(std::vector<double*> &signal) {
    // переписываем вектор входных сигналов
    for (int i = 0; i < inSignal.size(); i++) {
        inSignal[i] = *signal[i];
    }
    // расчитываем скалярное произведение
    for (int i = 0; i < inSignal.size(); i++) {
        outputValue += inSignal[i] * vectorOfWeights[i];
    }
    // применяем соответствующую функцию активации
    switch (activationFunction) {
        case 1:
            outputValue = logistic(outputValue, a);
            break;
            
            case 2:
            outputValue = identiti(outputValue, a, b);
            break;
            
            case 3:
            outputValue = hyperbolicTangent(outputValue, a, b);
            break;
    }
}

double& neuron::getE() {
    return e; // возвращаем ссылку на ошибку
}

double& neuron::theCalculationOfTheE(const double d) {
    e = d - outputValue;
    return e;
}

double &neuron::theCalculationOfTheLocalGradient() {
    // выбираем соответствующую производную функции активации
    switch (activationFunction) {
        case 1:
            localGradient = e * Dlogistic(outputValue, a);
            break;
            
            case 2:
            localGradient = e * Didentiti(outputValue, a, b);
            break;
            
            case 3:
            localGradient = e * DhyperbolicTangent(outputValue, a, b);
            break;
    }
    return localGradient;
}

void neuron::weightChangeCalculation(const double learningRate) {
    for (int i = 0; i < vectorOfWeights.size(); i++) {
        vectorOfWeights[i] += learningRate * localGradient * inSignal[i];
    }
}

neuron::neuron(const int TheSizeOfTheVectorOfWeights) { 
    vectorOfWeights.resize(TheSizeOfTheVectorOfWeights);
}

void neuron::theCalculationOfTheOutputValueFromInputSignal(std::vector<double> &signal) {
    // переписываем вектор входных сигналов
    for (int i = 0; i < inSignal.size(); i++) {
        inSignal[i] = signal[i];
    }
    // расчитываем скалярное произведение
    for (int i = 0; i < inSignal.size(); i++) {
        outputValue += inSignal[i] * vectorOfWeights[i];
    }
    // применяем соответствующую функцию активации
    switch (activationFunction) {
        case 1:
            outputValue = logistic(outputValue, a);
            break;
            
        case 2:
            outputValue = identiti(outputValue, a, b);
            break;
            
        case 3:
            outputValue = hyperbolicTangent(outputValue, a, b);
            break;
    }
}



double logistic(double x, double a, double b) {
    return b * (1 / (1 + exp(-a * x)));
}

double Dlogistic(double y, double a, double b) {
    return a * b * y * (1 - y);
}

double identiti(double x, double a, double b) {
    return a * x + b;
}

double Didentiti(double y, double a, double b) {
    return a;
}

double hyperbolicTangent(double x, double a, double b) {
    return a * ((exp(b * x) - exp(-b * x))/(exp(b * x) + exp(-b * x)));
}

double DhyperbolicTangent(double y, double a, double b) {
    return double(b / a) * ((a - y) * (a + y));
}

