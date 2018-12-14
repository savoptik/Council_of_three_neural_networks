//
//  layer.cpp
//  forward_direction_neural_network
//
//  Created by Артём Семёнов on 18/10/2018.
//  Copyright © 2018 Артём Семёнов. All rights reserved.
//

#include "layer.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>
using namespace std;

double layer::getError() { 
    return error; // ошибка на слое
}

std::vector<double *> &layer::AccessToTheErrorVector() { 
    return errors;
}

std::vector<double *> &layer::accessToTheLocalGradientVector() { 
    return localGradients;
}

std::vector<double *> &layer::AccessToTheOutputVector() { 
    return outputValues;
}

std::vector<neuron> &layer::accessToTheNeuronVector() { 
    return neurons;
}

layer::layer(const int number, const int size, const int af, const double a, const double b) {
    std::default_random_engine dra(static_cast<unsigned int>(time(0)));
    std::uniform_real_distribution<> urd(-0.8, 0.8);
    // создание нейронов
    for (int i = 0; i < number; i++) {
        neurons.push_back(neuron(size, af, a, b));
    }
    toRebuildThePointers(); // пересбор указателей
    ap = a;
    bp = b;
    one = urd(dra);
}

void layer::toCalculateTheOutputValuesForTheCurrentLayer(layer &previousLayer) {
    for (int i = 0; i < neurons.size(); i++) { // едем по нейронам
        neurons[i].theCalculationOfTheOutputValue(previousLayer.AccessToTheOutputVector()); // вычисляем выходные значения
    }
}

void layer::toCalculateTheComponentOfTheVectorOfErrors(vector<double>& d) { 
    for (int i = 0; i < neurons.size(); i++) {
        neurons[i].theCalculationOfTheE(d[i]); // вычисляем ошибку
        neurons[i].theCalculationOfTheLocalGradient(); // за одно и локальный градиент. только для выходного слоя
}
}

void layer::countTheWeightOnTheCurrentLayer(const double learningRate) { 
    for (int i = 0; i < neurons.size(); i++) {
        neurons[i].weightChangeCalculation(learningRate); // вычесляем изменения векторов весов
    }
}

void layer::calculateLocalGradientsForTheCurrentLayer(layer &previousLayer) { 
    for (int i = 0; i < localGradients.size(); i++) {
        double sum = 0;
            for (int k = 0; k < previousLayer.accessToTheNeuronVector().size(); k++) {
                sum += *previousLayer.accessToTheLocalGradientVector()[k] * previousLayer.accessToTheNeuronVector()[k].getVectorOfWeights()[i];
            }
        switch (neurons[0].getActivationFunction()) {
            case 1:
                *localGradients[i] = Dlogistic(*outputValues[i], ap) * sum;
                break;
                
                case 2:
                *localGradients[i] = Didentiti(*outputValues[i], ap, bp) * sum;
                break;
                
                case 3:
                *localGradients[i] = DhyperbolicTangent(*outputValues[i]) * sum;
                break;
        }
    }
}

void layer::toCalculateTheError() {
    double sum = 0;
    for (int i = 0; i < errors.size(); i++) {
        sum += pow(*errors[i], 2);
    }
    error = sum / 2;
}

void layer::toRebuildThePointers() { 
    errors.resize(neurons.size());
    outputValues.resize(neurons.size());
    localGradients.resize(neurons.size());
    for (int i = 0; i < outputValues.size(); i++) {
        outputValues[i] = &neurons[i].getOutputValue();
        localGradients[i] = &neurons[i].getLocalGradient();
        errors[i] = &neurons[i].getE();
    }
    outputValues.push_back(&one);
}

layer::layer(const int NumberOfNeurons, const int TheSizeOfTheVectorOfWeights) {
    std::default_random_engine dra(static_cast<unsigned int>(time(0)));
    //    std::mt19937 gen(rd.entropy());
    std::uniform_real_distribution<> urd(-0.8, 0.8);
    neurons.resize(NumberOfNeurons);
    for (int i = 0; i < neurons.size(); i++) {
        neurons[i] = neuron(TheSizeOfTheVectorOfWeights);
    }
    toRebuildThePointers();
    one = urd(dra);
}

void layer::calculateTheOutputValuesByTheVectorOfTheInputSignals(std::vector<double> &inputSignals) { 
    for (int i = 0; i < neurons.size(); i++) {
        neurons[i].theCalculationOfTheOutputValueFromInputSignal(inputSignals);
    }
}
