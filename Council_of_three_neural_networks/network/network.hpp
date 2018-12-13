//
//  network.hpp
//  forward_direction_neural_network
//
//  Created by Артём Семёнов on 22/10/2018.
//  Copyright © 2018 Артём Семёнов. All rights reserved.
//

#ifndef network_hpp
#define network_hpp

#include <iostream>
#include "./layer/layer.hpp"

class network {
public:
    network(std::vector<int>& sizes, const int num, const int af, const double ap, const double bp=0);
    network(const std::string filePath);
    /**
     Доступ к вектору слоёв

     @return ссылку на верктор слоёв
     */
    std::vector<layer>& accessToLayers();
    std::vector<double *> *accessToOutVector() {return outVector;}
    /**
     Доступ к вектору ошибок

     @return ссылку на вектор ошибок
     */
    std::vector<double>& accessToErrors();
    /**
     метод экспортирует сеть в заданную папку

     @param folderPath путь к папке
     */
    void exportNetwork(const std::string folderPath);
    /**
     Обучение

     @param TrainingSample набор векторов входных сигналов
     @param response вектор векторов откликов
     @param epsErrorse порог ошибки
     @param lg локальный градиент
     @param MaximumNumberOfEpochs максимальное количество эпох обучения по умолчанию 100
     */
    void train(std::vector<std::vector<double> > &TrainingSample, std::vector<std::vector<double>> &response, const double epsErrorse, const double lg, const int MaximumNumberOfEpochs);
    /**
     Прямой проход

     @param inputSignal входной сигнал
     */
    void directPropagation(std::vector<double> &inputSignal);
private:
    /**
     Обратный проход

     @param responseVector вектор откликов
     @param lg локальный градиент
     */
    void backPropagation(std::vector<double> &responseVector, const double lg);
    void shuffleIndexes(std::vector<int> &indexes);
    std::vector<layer> layers;
    std::vector<double> errors;
    int nans = 0;
    std::vector<double *> *outVector;
};

#endif /* network_hpp */
