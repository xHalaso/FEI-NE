import numpy as np
import torch
import torch.nn as nn
import networks as net
import genetic_all as ga

class genome():
    # todo read mutation and etc from config as list/dict 
    def __init__(self, model,config,debug): 
        self.genomes = np.array([]) 
        self.NumOfGenomes = 0
        self.model = model
        self.debug = debug
        
        self.crossConfig = config.get("crossConfig")
        self.wRange = config.get("weightRange")
        self.popsize = config.get("popSize")
        
        if(self.debug):
            print("\n", 
                  f"Crossover Config: {self.crossConfig}\n",
                  f"ANN Model: {self.model}\n" )
         
    def generPop(self):
        #
        self.genomes = self.model.getWeight2Array() 
        for gen in range(self.popsize-1):
            self.model.generGenomeWeight(self.wRange)
            self.genomes = np.vstack((self.genomes,self.model.getWeight2Array()) )
        
        if self.debug: print(self.genomes)
    
    def selection(self):
        pass
    
    def crossGenomes(self):
        if self.debug: oldpop = np.copy(self.genomes)
            
        if (self.crossConfig.get("typeOfCross") == "crossov"):
            ga.crossov(self.genomes,self.crossConfig.get("split_points"),self.crossConfig.get("model"))
        elif (self.crossConfig.get("typeOfCross") == "crossgrp"):
            ga.crossgrp(self.genomes,self.crossConfig.get("split_points"))
        else:
            ("typeOfCross ERROR")
        
        if self.debug: print("Compare after cross:\n",oldpop == self.genomes)
        
    def localMut(self):
        pass
    
    def globalMut(self):
        pass