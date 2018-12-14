//
//  neuron.hpp
//  forward_direction_neural_network
//
//  Created by Admin on 11/10/2018.
//  Copyright © 2018 Артём Семёнов. All rights reserved.
//

#ifndef neuron_hpp
#define neuron_hpp

#include <stdio.h>
#include <vector>

class neuron {
public:
    neuron() {}
    neuron(const int TheSizeOfTheVectorOfWeights); // конструктор создаёт нейрон с пустым вектором весов заданного размера
    neuron(int size, const int acFunc, const double afa, const double afb = 0);
    /**
     вектор весов

     @return ссылку на вектор весов
     */
    std::vector<double>& getVectorOfWeights();
    /**
     номер функции активации для этого нейрона

     @return номер функции активации
     */
    int getActivationFunction();
    /**
     локальный градиент

     @return локальный градиент
     */
    double& getLocalGradient();
    /**
     выходное значение

     @return выходное значение
     */
    double& getOutputValue();
    /**
     вектор входных сигналов

     @return ссылку на вектор входных сигналов
     */
    std::vector<double>& getInputVector();
    /**
     Расчёт выходного значений

     @param signal вектор входного сигнала
     */
    void theCalculationOfTheOutputValue(std::vector<double*>& signal);
    void theCalculationOfTheOutputValueFromInputSignal(std::vector<double>& signal);
    /**
     Расчёт локального градиента

     @return ссылку на значение поля локального градиента
     */
    double& theCalculationOfTheLocalGradient();
    /**
     Расчёт компонента вектора ошибки

     @param d компонент вектора отклика
     @return ссылку на поле компонента вектора ошибки
     */
    double& theCalculationOfTheE(const double d);
    /**
     Доступ к полю компонента вектора ошибки

     @return ссылку на компонент вектора ошибки
     */
    double& getE();
    /**
     Пересчёт вектора весов

     @param learningRate Скорость обучения
     */
    void weightChangeCalculation(const double learningRate);
private:
    std::vector<double> vectorOfWeights; // вектор весов
    std::vector<double> inSignal; // сохранённый входной сигнал
    int activationFunction = 0; // номер функции активации
    double localGradient = 0.0; // локальный градиент
    double outputValue = 0.0; // выходное значение
    double a = 0.0, b = 0.0; // параметры функции активации
    double e = 0.0; //компонент вектора ошибки
};

// функции активации и их производные
double logistic(double x, double a = 1, double b = 1); // логистическая
double Dlogistic(double y, double a = 1, double b = 1); // производная логистической функции
double identiti(double y, double a = 1, double b = 0); // функция идентичности
double Didentiti(double y, double a = 1, double b = 0); // производная функции идентичности
double hyperbolicTangent(double x, double a = 1.7, double b = double(2/3)); // функция гиперболического тангенса
double DhyperbolicTangent(double y, double a = 1.7, double b = double(2/3)); // производна функции гиперболического тангенса

#endif /* neuron_hpp */
