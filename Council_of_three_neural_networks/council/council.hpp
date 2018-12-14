//
//  council.hpp
//  Council_of_three_neural_networks
//
//  Created by Артём Семёнов on 13/12/2018.
//  Copyright © 2018 Артём Семёнов. All rights reserved.
//

#ifndef council_hpp
#define council_hpp

#include "network/network.hpp"

const int EAGLE = 1, TAILS = 0; // орёл и решка

class council {
    network MrFirst; // первый эксперт
    network MrSecond; // второй эксперт
    network decisiveExpert; // решающий эксперт
    /**
     Функция подбрасывает монетку

     @return Возвращает 1 в случае орла, или 0 в случае решки
     */
    int flipACoin();
    
public:
    council() {}
    council(std::vector<int>& numberOfNuerons, const int sizeOfVectorOfWeights); // конструктор, который сконструирует всех трёх экспертов
    /**
     Запуск обучения

     @param trainSimples вектор примеров
     @param lables вектор откликов
     */
    void train(std::vector<std::vector<double>>& trainSimples, std::vector<std::vector<double>>& lables);
};

#endif /* council_hpp */
