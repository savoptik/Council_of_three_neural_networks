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

class council {
    network MrFirst; // первый эксперт
    network MrSecond; // второй эксперт
    network decisiveExpert; // решающий эксперт
    
public:
    council() {}
    council(std::vector<int>& numberOfNuerons, const int sizeOfVectorOfWeights); // конструктор, который сконструирует всех трёх экспертов
};

#endif /* council_hpp */
