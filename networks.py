import torch
import torch.nn as nn
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self,config):
        super(NeuralNetwork, self).__init__()
        self.createNN(config.get("in"),config.get("out"),config.get("hidden"))

    def createNN(self, input_size, output_size, hidden_layers):
        modules = []
        last_size = input_size

        # Vytvorenie skrytých vrstiev
        for size in hidden_layers:
            modules.append(nn.Linear(last_size, size,bias=False))
            last_size = size

        # Pridanie výstupnej vrstvy
        modules.append(nn.Linear(last_size, output_size,bias=False))

        # Vytvorenie sekvencie modulov
        self.layers = nn.Sequential(*modules)
        
    def forward(self, x):
        if not self.layers:
            raise Exception("Neurónová sieť nebola vytvorená. Použite metódu createNN.")
        x = torch.tanh(x)                           # Input Nodes
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:            # Hidden Nodes
                x = torch.tanh(x)
            else:                                   # Output Nodes
                x = torch.tanh(x)
        return x
    
        # zeros initialization method 
        # torch.nn.init.zeros_(linear_layer.weight) 
        
    def generGenomeWeight(self,rangeW):
        # User defined function to initialize the weights
        if not self.layers:
            raise Exception("Neurónová sieť nebola vytvorená. Použite metódu createNN.")
        for i, layer in enumerate(self.layers):
            torch.nn.init.uniform_(layer.weight,-rangeW, rangeW) 
            
    # def getWeight2(self):
    #     for name, param in self.named_parameters():
    #         if "weight" in name:  
    #             print(f"Original weight of {name}: {param.data}")
    #             param.data.fill_(1.0)  
    #             print(f"Modified weight of {name}: {param.data}")
    
    def getWeight2Array(self):
        # 1. Extrahovanie váh do NumPy polí
        weights = np.array([]) 
        for param in self.parameters():
            weights = np.append(weights,torch.flatten(param.data).numpy()) 
            #print ("Flatten",torch.flatten(param.data).numpy())
        return weights
      
        
# Todo def for apply weight after GA from genome class        
# # User defined function to initialize the weights 
# def custom_weights(m): 
#     torch.nn.init.uniform_(m.weight, 
#                            -0.5, 0.5) 
  
# # Initializing a linear layer with  
# # 2 independent features and 3 dependent features 
# linear_layer = torch.nn.Linear(2, 3) 
  
# # Applying the user defined function to the layer 
# linear_layer.apply(custom_weights) 
  
# # Displaying the initialized weights 
# print(linear_layer.weight) 



    # nn_model.getWeight()

    # output = nn_model(torch.tensor([5,5,5,5]).float())  
    # print(output)
