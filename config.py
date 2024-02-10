class configLoad():
    def __init__(self):
        self.crossConfig = dict(
            typeOfCross = "crossov", 
            split_points = 3, 
            mode = 0)
        self.mutationConfig = {
            'global'    : {"func": "mutx" ,"rate": 0.1},
            'local'     : {"func": "muta" ,"rate": 0.15, "ampdelimiter":100},
        }
        self.subPop1 = { 
                    'crossConfig'       : self.crossConfig,
                    'mutationConfig'    : self.mutationConfig,
                    'popSize'           : 5,
                    'weightRange'       : 5
                    }

        self.modelConfig = { 
                        'in'            : 1,
                        'out'           : 2,
                        'hidden'        : [2,2],
                        'act'           : "tanh"
                        }
    def getSubPops(self):
        return self.subPop1
    def getModel(self):
        return self.modelConfig