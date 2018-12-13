//
//  layer.hpp
//  forward_direction_neural_network
//
//  Created by Артём Семёнов on 18/10/2018.
//  Copyright © 2018 Артём Семёнов. All rights reserved.
//

#ifndef layer_hpp
#define layer_hpp

#include <stdio.h>
#include "../neuron/neuron.hpp"

class layer {
public:
    layer() {} // конструктор бес параметров
    layer(const int NumberOfNeurons, const int TheSizeOfTheVectorOfWeights); // конструктор, создающий нейросеть с заданными размерами
    layer(const int number, const int size, const int af, const double a, const double b = 0); // конструктор, принимающий количество нейронов, которые нужно создать, и создающий это количество неййронов.
    /**
     Доступ к вектору нейронов

     @return ссылку на вектор нейронов
     */
    std::vector<neuron>& accessToTheNeuronVector();
    /**
     доступ к вектору выходных значений

     @return ссылку на вектор выходных значений
     */
    std::vector<double*>& AccessToTheOutputVector();
    /**
     Доступ к вектору локальных градиентов

     @return ссылку на вектор локальных градиентов
     */
    std::vector<double*>& accessToTheLocalGradientVector();
    /**
     Доступ к вектору ошибок

     @return ссылку на вектор ошибок
     */
    std::vector<double*>& AccessToTheErrorVector();
    /**
     Получить ошибку слоя

     @return ошибку
     */
    double getError();
    /**
     Запускает процесс вычисления выходного значения

     @param previousLayer ссылку на объект предыдущего слоя
     */
    void toCalculateTheOutputValuesForTheCurrentLayer(layer& previousLayer);
    /**
     Метод запускает расчёт компонентов вектора ошибки

     @param d компонент вектора отклика
     */
    void toCalculateTheComponentOfTheVectorOfErrors(std::vector<double> &d);
    /**
     разчёт локальных градиентов по данным с предыдущего слоя

     @param previousLayer указатель на предыдущий слой
     */
    void calculateLocalGradientsForTheCurrentLayer(layer& previousLayer);
    /**
     Вычислить выходные значения по вектору входных сигналов

     @param inputSignals вектор входных сигналов
     */
    void calculateTheOutputValuesByTheVectorOfTheInputSignals(std::vector<double> &inputSignals);
    /**
     метод запускает пересчёт весов

     @param learningRate скорость обучения
     */
    void countTheWeightOnTheCurrentLayer(const double learningRate);
    /**
     Вектор вычисляет ошибку слоя
     */
    void toCalculateTheError();
    /**
     Метод пересобирает указатели
     */
    void toRebuildThePointers();
private:
    std::vector<neuron> neurons; // набор нейронов
    std::vector<double *> outputValues; // вектор указателй на выходные значения
    std::vector<double *> localGradients; // вектор указателей на локальные градиенты
    std::vector<double *> errors; // вектор указателей на компоненты вектора ошибок.
    double error = 0; // Ошибка
    double ap = 0, bp = 0;
    double one = 1;
};

#endif /* layer_hpp */
