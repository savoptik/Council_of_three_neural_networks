//
//  council.cpp
//  Council_of_three_neural_networks
//
//  Created by Артём Семёнов on 13/12/2018.
//  Copyright © 2018 Артём Семёнов. All rights reserved.
//

#include "council.hpp"
#include <iostream>
#include <random>
#include <algorithm>
#define numberOfSimple (5000)
using namespace std;

council::council(std::vector<int>& numberOfNuerons, const int sizeOfVectorOfWeights) {
    MrFirst = network(numberOfNuerons, sizeOfVectorOfWeights, 1, 1);
    MrSecond = network(numberOfNuerons, sizeOfVectorOfWeights, 1, 1);
    decisiveExpert = network(numberOfNuerons, sizeOfVectorOfWeights, 1, 1);
    cout << "первый эксперт имеет в своём составе " << MrFirst.accessToLayers().size() << " слоёв размерами: ";
    for (int i = 0; i < MrFirst.accessToLayers().size(); i++) {
        cout << MrFirst.accessToLayers()[i].accessToTheNeuronVector().size() << " ";
    } cout << endl;
    cout << "Второй эксперт имеет в своём составе " << MrSecond.accessToLayers().size() << " слоёв размерами: ";
    for (int i = 0; i < MrSecond.accessToLayers().size(); i++) {
        cout << MrSecond.accessToLayers()[i].accessToTheNeuronVector().size() << " ";
    } cout << endl;
    cout << "Решающий эксперт имеет в своём составе " << MrFirst.accessToLayers().size() << " слоёв размерами: ";
    for (int i = 0; i < decisiveExpert.accessToLayers().size(); i++) {
        cout << decisiveExpert.accessToLayers()[i].accessToTheNeuronVector().size() << " ";
    } cout << endl;
}

void council::train(std::vector<std::vector<double> > &trainSimples, std::vector<std::vector<double> > &lables) { 
    vector<vector<double>> filtrTrainingSimples; // вектор для выборки обучения первой сети
    vector<vector<double>> filtrLables; // набор меток для обучения первой сети
    for (int i  = 0; i < numberOfSimple; i++) { // забираем из общей выборки 10000 примеров
        filtrTrainingSimples.push_back(trainSimples[i]);
        filtrLables.push_back(lables[i]);
    }
    cout << " Удаляю из выборки ненужные элементы, осталось ";
    trainSimples.erase(trainSimples.begin(), trainSimples.begin()+numberOfSimple);
    lables.erase(lables.begin(), lables.begin()+numberOfSimple);
    cout << trainSimples.size() << " примеров и " << lables.size() << " меток\n";
    cout << "тренирую первую сеть  на " << filtrTrainingSimples.size() << " примерах и " << filtrLables.size() << " метках\n";
    toCollectStatistics(filtrLables);
    MrFirst.train(filtrTrainingSimples, filtrLables, 0.01, 0.6, 100);
    cout << "Отбираем примеры для обучения второй сети:\n";
    int i = 0;
    filtrLables.clear();
    filtrTrainingSimples.clear();
    vector<int> indexes;
    int coin = flipACoin();
    while ((filtrTrainingSimples.size() < numberOfSimple)) {
        MrFirst.directPropagation(trainSimples[i]);
        if (coin == EAGLE) {
            if (theTransformationOfTheVectorOfOutputSignals(lables[i]) == theTransformationOfTheVectorOfOutputSignalsP(*MrFirst.accessToOutVector())) {
                indexes.push_back(i);
                filtrTrainingSimples.push_back(trainSimples[i]);
                filtrLables.push_back(lables[i]);
                coin = flipACoin();
            }
        } else if (coin == TAILS) {
            if (theTransformationOfTheVectorOfOutputSignals(lables[i]) != theTransformationOfTheVectorOfOutputSignalsP(*MrFirst.accessToOutVector())) {
                indexes.push_back(i);
                filtrTrainingSimples.push_back(trainSimples[i]);
                filtrLables.push_back(lables[i]);
                coin = flipACoin();
            }
        }
        i++;
        i = i == trainSimples.size()? 0: i;
    }
    vector<vector<double>> timeTrainingSimples, timeLables;
    for (i = 0; i < trainSimples.size(); i++) {
        if (inVector(indexes, i) == false) {
            timeTrainingSimples.push_back(trainSimples[i]);
            timeLables.push_back(lables[i]);
        }
    }
    trainSimples.clear();
    lables.clear();
    cout << "Осталось в основной выборке " << timeTrainingSimples.size() << " примеров и " << timeLables.size() << " меток\n";
    cout << "Обучаю вторую сеть на " << filtrTrainingSimples.size() << " примерах и " << filtrLables.size() << " метках\n";
toCollectStatistics(filtrLables);
    MrSecond.train(filtrTrainingSimples, filtrLables, 0.01, 0.6, 100);
    cout << "Готовлю выборку для обучения решающего эксперта:\n";
    filtrTrainingSimples.clear();
    filtrLables.clear();
    i = 0;
    while ((filtrTrainingSimples.size() < numberOfSimple) && (i < timeTrainingSimples.size())) {
        MrFirst.directPropagation(timeTrainingSimples[i]);
        MrSecond.directPropagation(timeTrainingSimples[i]);
        if (theTransformationOfTheVectorOfOutputSignalsP(*MrFirst.accessToOutVector()) != theTransformationOfTheVectorOfOutputSignalsP(*MrSecond.accessToOutVector())) {
            filtrTrainingSimples.push_back(timeTrainingSimples[i]);
            filtrLables.push_back(timeLables[i]);
        }
        i++;
    }
    cout << "Обучаю решающую сеть на " << filtrTrainingSimples.size() << " примерах и " << filtrLables.size() << " метках\n";
    toCollectStatistics(filtrLables);
    decisiveExpert.train(filtrTrainingSimples, filtrLables, 0.01, 0.6, 100);
}

int council::flipACoin() { 
    default_random_engine dra(static_cast<unsigned int>(time(0)));
    uniform_int_distribution<> urd(0, 1);
    return urd(dra);
}

int council::predict(std::vector<double> &example) { 
    MrFirst.directPropagation(example);
    int firstResponse = theTransformationOfTheVectorOfOutputSignalsP(*MrFirst.accessToOutVector());
    MrSecond.directPropagation(example);
    int secondResponse = theTransformationOfTheVectorOfOutputSignalsP(*MrSecond.accessToOutVector());
    if (firstResponse == secondResponse) {
        return firstResponse;
    } else {
        decisiveExpert.directPropagation(example);
        return theTransformationOfTheVectorOfOutputSignalsP(*decisiveExpert.accessToOutVector());
    }
}

void council::toCollectStatistics(std::vector<std::vector<double> > &lables) { 
    std::vector<int> numbers;
    numbers.resize(10);
    for (int i = 0; i < lables.size(); i++) {
        numbers[theTransformationOfTheVectorOfOutputSignals(lables[i])]++;
    }
    for (int i = 0; i < numbers.size(); i++) {
        std::cout << i << ": " << numbers[i] << std::endl;
    }
}



void convertingLabels(std::vector<u_char>& inLables, std::vector<std::vector<double>>& lables) {
    // преобразуем вектор откликов к виду, в котором еденица означает число
    lables.resize(inLables.size());
    for (long i = 0; i < lables.size(); i++) { // обходим вектор
        lables[i].resize(10); // задаём размер
        lables[i][inLables[i]] = 1; // записываем на соответствующую позицию единицу
    }
}

void imageConversion(std::vector<std::vector<u_char>>& charImages, std::vector<std::vector<double>>& outImages) {
    outImages.resize(charImages.size());
    for (int i  = 0; i < outImages.size(); i++) {
        outImages[i].resize(charImages[i].size());
        for (int j = 0; j < outImages[i].size(); j++) {
            outImages[i][j] = charImages[i][j];
        }
        outImages[i].push_back(1);
    }
}

int theTransformationOfTheVectorOfOutputSignals(std::vector<double>& outputSignal) {
    int result = 0;
    double max = 0;
    for (int i = 0; i < outputSignal.size(); i++) {
        if (outputSignal[i] > max) {
            max = outputSignal[i];
            result = i;
        }
    }
    return result;
}

int theTransformationOfTheVectorOfOutputSignalsP(std::vector<double *>& outputSignal) {
    int result = 0;
    double max = 0;
    for (int i = 0; i < outputSignal.size()-1; i++) {
        if (*outputSignal[i] > max) {
            max = *outputSignal[i];
            result = i;
        }
    }
    return result;
}

bool inVector(std::vector<int>& vec, const int n) {
    bool response = false;
    int i = 0;
    while ((response == false) && (i < vec.size())) {
        response = n == vec[i]? true: false;
        i++;
    }
    return response;
}
