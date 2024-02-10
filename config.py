class configLoad():
    def __init__(self):
        self.crossConfig = dict(
            typeOfCross = "crossov", 
            split_points = 3, 
            mode = 0)

        self.subPop1 = { 
                    'crossConfig'   : self.crossConfig,
                    'popSize'       : 5,
                    'weightRange'   : 5
                    }

        self.modelConfig = { 
                        'in'            : 1,
                        'out'           : 2,
                        'hidden'        : [2],
                        'act'           : 'tanh'
                        }
    def getSubPops(self):
        return self.subPop1
    def getModel(self):
        return self.modelConfig