import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layers = None

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

    # def forward(self, x):
    #     if self.layers:
    #         for layer in self.layers:
    #             x = nn.Tanh(layer(x))
    #         return x
    #         # return self.layers(x)
    #     else:
    #         raise Exception("Neurónová sieť nebola vytvorená. Použite metódu createNN.")
        
    def forward(self, x):
        if not self.layers:
            raise Exception("Neurónová sieť nebola vytvorená. Použite metódu createNN.")
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Aplikuj tanh na všetky vrstvy okrem poslednej
                x = torch.tanh(x)
            else:  # Na poslednú vrstvu tiež aplikuj tanh
                x = torch.tanh(x)
        return x
    
        
    def getWeight2(self):
        for name, param in self.named_parameters():
            if "weight" in name:  # Kontrola, či je parameter váha
                # Tu môžete váhy upraviť, napríklad vynulovaním alebo nastavením na konkrétne hodnoty
                print(f"Original weight of {name}: {param.data}")
                param.data.fill_(1.0)  # Nastaví všetky váhy na 1.0
                print(f"Modified weight of {name}: {param.data}")
    
    def getWeight(self):
        # weights = [param.data for param in self.parameters()]
        # #print(weights)

        # for w in weights:
        #     print(w,"\n")

        # 1. Extrahovanie váh do NumPy polí
        weights = []
        for param in self.parameters():
            weights.append((param.data.numpy())[0])  # Konvertujte PyTorch tensor na NumPy pole
        print("weight: ", weights)

        # 2. Modifikácia váh
        # Tento krok závisí od vašich potrieb. Ako príklad, tu je jednoduchá modifikácia:
        modified_weights = [w *0 for w in weights]  # Pridanie 1 ku každému prvku v každom poli váh    
        print("new weight: ", modified_weights)



if __name__ == '__main__':
    # Použitie triedy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nn_model = NeuralNetwork().to(device)
    
    nn_model.createNN(1, 2, [2])  # Príklad: 4 vstupy, 2 výstupy, skryté vrstvy [3, 10, 5]
    print(nn_model)
    nn_model.getWeight()

    # output = nn_model(torch.tensor([5,5,5,5]).float())  
    # print(output)








    #print(nn.Parameter(weights, requires_grad=False))
    # for layer in nn_model.layers:
    #     print(f"Layer shape:{layer.weight.shape}")
        # weights = layer.weight
        # array=weights.detach().numpy().shape
        # print(f"Layer:{array}")
