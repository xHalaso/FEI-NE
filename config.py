class configLoad():
    #todo load from file
    def __init__(self):
        self.crossConfig = dict(
            typeOfCross = "crossov", 
            split_points = 3, 
            mode = 0)
        self.mutationConfig = {
            'global'    : {"func": "mutx" ,"rate": 0.1},
            'local'     : {"func": "muta" ,"rate": 0.15, "ampdelimiter":100},
        }
        self.Pop = { 
                    'crossConfig'       : self.crossConfig,
                    'mutationConfig'    : self.mutationConfig,
                    'popSize'           : 5,
                    'weightRange'       : 5,
                    'elit'              : 10,
                    'sels'              : 80,
                    'seltourn'          : 10
                    }

        self.modelConfig = { 
                        'in'            : 1,
                        'out'           : 2,
                        'hidden'        : [2,2],
                        'act'           : "tanh"
                        }
    def getPops(self):
        return self.Pop
    def getModel(self):
        return self.modelConfig