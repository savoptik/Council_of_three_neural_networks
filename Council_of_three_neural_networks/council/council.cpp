//
//  council.cpp
//  Council_of_three_neural_networks
//
//  Created by Артём Семёнов on 13/12/2018.
//  Copyright © 2018 Артём Семёнов. All rights reserved.
//

#include "council.hpp"
#include <iostream>
using namespace std;

council::council(std::vector<int>& numberOfNuerons, const int sizeOfVectorOfWeights) {
    MrFirst = network(numberOfNuerons, sizeOfVectorOfWeights, 1, 1);
    MrSecond = network(numberOfNuerons, sizeOfVectorOfWeights, 1, 1);
    decisiveExpert = network(numberOfNuerons, sizeOfVectorOfWeights, 1, 1);
    cout << "первый эксперт имеет в своём составе " << MrFirst.accessToLayers().size() << " размерами: ";
    for (int i = 0; i < MrFirst.accessToLayers().size(); i++) {
        cout << MrFirst.accessToLayers()[i].accessToTheNeuronVector().size() << " ";
    } cout << endl;
    cout << "Второй эксперт имеет в своём составе " << MrSecond.accessToLayers().size() << " размерами: ";
    for (int i = 0; i < MrSecond.accessToLayers().size(); i++) {
        cout << MrSecond.accessToLayers()[i].accessToTheNeuronVector().size() << " ";
    } cout << endl;
    cout << "Решающий эксперт имеет в своём составе " << MrFirst.accessToLayers().size() << " размерами: ";
    for (int i = 0; i < decisiveExpert.accessToLayers().size(); i++) {
        cout << decisiveExpert.accessToLayers()[i].accessToTheNeuronVector().size() << " ";
    } cout << endl;
}

void council::train(std::vector<std::vector<double> > &trainSimples, std::vector<std::vector<double> > &lables) { 
    vector<vector<double>> firstTrainingSimples; // вектор для выборки обучения первой сети
    vector<vector<double>> firstLables; // набор меток для обучения первой сети
    for (int i  = 0; i < 10000; i++) { // забираем из общей выборки 10000 примеров
        firstTrainingSimples.push_back(trainSimples[i]);
        firstLables.push_back(lables[i]);
    }
    cout << "тренирую первую сеть \n";
    MrFirst.train(firstTrainingSimples, firstLables, 0.01, 0.6, 100);
    cout << " Удаляю из выборки ненужные элементы, осталось ";
        trainSimples.erase(trainSimples.begin(), trainSimples.begin()+9999);
    lables.erase(lables.begin(), lables.begin()+9999);
    cout << trainSimples.size() << " примеров и " << lables.size() << " меток\n";
}

