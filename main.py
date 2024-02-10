import config as conf
import networks as nn
import genome as g

if __name__ == '__main__':
    #todo load cofing for ga and etc 
    configs = conf.configLoad()
    nn_model = nn.NeuralNetwork(configs.getModel(),True)
    
    #nn_model.createNN(1, 2, [2])  # Príklad: 4 vstupy, 2 výstupy, skryté vrstvy [3, 10, 5]
    
    pop = g.genome(nn_model,configs.getSubPops(),False)
    pop.generPop()
       
    pop.crossGenomes()
