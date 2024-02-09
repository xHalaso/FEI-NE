import torch
import torch.nn as nn
import numpy as np
import networks as nn
import genetic_all as ga
import genome as g

crossConfig = dict(
    typeOfCross = "crossov", 
    split_points = 3, 
    mode = 0)

subPop1 = { 'crossConfig'   : crossConfig,
            'popSize'       : 5,
            'weightRange'   : 5
            }

modelConfig = { 'in'            : 1,
                'out'           : 2,
                'hidden'        : [2],
                'act'           : 'tanh'
                }

if __name__ == '__main__':
    #todo load cofing for ga and etc
    # Použitie triedy
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nn_model = nn.NeuralNetwork(modelConfig).to(device)
    
    #nn_model.createNN(1, 2, [2])  # Príklad: 4 vstupy, 2 výstupy, skryté vrstvy [3, 10, 5]
    
    pop = g.genome(nn_model,subPop1,False)
    pop.generPop()
       
    pop.crossGenomes()
