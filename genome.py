import numpy as np
import torch
import torch.nn as nn
import networks as net
import genetic_all as ga
import itertools

class genome():
    # todo div to genome(only genome info) and Pop(gener genomes, Ga operation etc) 
    id_iter = itertools.count()
    
    def __init__(self, model,weightRange): 
        self.id = next(self.id_iter)
        self.model = model
        self.gen    = None
        self.fitness = None
        self.debug = False
        self.regenWeight(weightRange)

                 
    def regenWeight(self,weightRange=5):
        self.model.generGenomeWeight(weightRange)
        self.gen = np.array(self.model.getWeight2Array())
            
    def calStep():
        pass