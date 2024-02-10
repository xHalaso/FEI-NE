import numpy as np
import genome as g
import genetic_all as ga

class Population():
    def __init__(self,config,debug,debugL):
        self.crossConfig = config.get("crossConfig")
        self.mutationConfig = config.get("mutationConfig")
        self.debug = debug
        self.debugL = debugL
        self.wRange = config.get("weightRange")
        self.popsize = config.get("popSize")
        self.genomes = np.array([])
        self.space = None
        self.amp = None
        
        if((self.debug) and (self.debugL>=2)):
            print(f"Pop Config: {config}\n")
    
    def generPop(self, model):
        self.genomes = np.array([g.genome(model, self.wRange) for _ in range(self.popsize)], dtype=object)
        self.createSpaceAmp()
        if ((self.debug) and (self.debugL==4)):
            for i,geno in enumerate(self.genomes):
                print(f"{i}. {geno.gen}\n {geno.model}")
            print(f"AMP: {self.amp} \n Space: {self.space}")
        elif((self.debug) and (self.debugL==3)):
            for i,geno in enumerate(self.genomes):
                print(f"{i}. {geno.gen}\n")
        elif((self.debug) and (self.debugL==2)):
            print(f"Size of Pop: {self.genomes.size} , Number of gens: {self.genomes[0].gen.size}\n")
    
    def createSpaceAmp(self):
        if self.genomes[0].gen.size != None:
            self.space = np.array([np.ones(self.genomes[0].gen.size) * (-self.wRange), np.ones(self.genomes[0].gen.size) * self.wRange])
            self.amp = self.space[1, :] / self.mutationConfig.get("local").get("ampdelimiter")
        
    def selection(self, newpop):
        pass
    
    def crossGenomes(self, newpop):
        if (self.crossConfig.get("typeOfCross") == "crossov"):
            ga.crossov(newpop,self.crossConfig.get("split_points"),self.crossConfig.get("model"))
        elif (self.crossConfig.get("typeOfCross") == "crossgrp"):
            ga.crossgrp(newpop,self.crossConfig.get("split_points"))
        else:
            ("typeOfCross ERROR")

    def Mut(self, newpop):
        ga.mutx(newpop,self.mutationConfig.get("global").get("rate"),self.space)
        ga.muta(newpop,self.mutationConfig.get("local").get("rate"),self.amp,self.space)

        
    def gaLoop(self):
        #todo making subPops via selections
        newpop=np.array([gen.gen for gen in self.genomes])
        if ((self.debug) and (self.debugL>=2)): oldpop=np.copy(newpop); print(oldpop)
        
        # self.selection(newpop)
        self.crossGenomes(newpop)
        self.Mut(newpop)
        
        for gen, new_gen_values in zip(self.genomes, newpop):
            gen.gen = new_gen_values
        
        if ((self.debug) and (self.debugL>=2)): print("Compare after GlobalMut:\n", oldpop == np.array([gen.gen for gen in self.genomes]),f"\n {np.array([gen.gen for gen in self.genomes])}")
        pass    