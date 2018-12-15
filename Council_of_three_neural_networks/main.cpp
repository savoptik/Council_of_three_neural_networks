//
//  main.cpp
//  Council_of_three_neural_networks
//
//  Created by Артём Семёнов on 13/12/2018.
//  Copyright © 2018 Артём Семёнов. All rights reserved.
//

#include <iostream>
#include "council/council.hpp"
#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"
using namespace std;

int main(int argc, const char * argv[]) {
    auto ds = mnist::read_dataset();
    vector<int> numbersOfNuerons = {400, 10};
    cout << "Читаю базу MNIST\n";
    mnist::binarize_dataset(ds);
    cout << "прочитано " << ds.training_images.size() << " тренеровочных примеров, " << ds.training_labels.size() << " меток, " << ds.test_images.size() << " тестовых примеров и " << ds.test_labels.size() << " меток.\n";
    vector<vector<double>> convertResponse;
    vector<vector<double>> trainData;
    imageConversion(ds.training_images, trainData);
    convertingLabels(ds.training_labels, convertResponse);
    cout << "После преобразования " << trainData.size() << " примеров и " << convertResponse.size() << " меток.\n";
    cout << "Готовлю сеть:\n";
    council couns = council(numbersOfNuerons, static_cast<int>(ds.training_images[0].size()));
    couns.train(trainData, convertResponse);
    vector<vector<double>> testSimples, testLables;
    imageConversion(ds.test_images, testSimples);
    convertingLabels(ds.test_labels, testLables);
    int good = 0;
    for (int i = 0; i < testSimples.size(); i++) {
        good = couns.predict(testSimples[i]) == theTransformationOfTheVectorOfOutputSignals(testLables[i])? good+1: good;
    }
    cout << "Доля успешных предсказаний " << double(good) / testSimples.size() << endl << "всё\n";
    return 0;
}
