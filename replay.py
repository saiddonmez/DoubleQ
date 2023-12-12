import numpy as np

class replayBuffer():

    def __init__(self,size,NUMACTIONS):
        self.dataset = {'state':np.zeros((size,4,84,84),dtype=np.int8),
                        'action':np.zeros(size,dtype=np.int16),
                        'reward':np.zeros(size,dtype=np.float32),
                        'nextState':np.zeros((size,4,84,84),dtype=np.int8),
                        'terminated':np.zeros(size,dtype=np.bool)
                        }
        
        self.lastIndex = 0
        self.size = size
        self.full = False
        
    def recordSample(self,state,action,reward,nextState,terminated):
        
        if (self.lastIndex == self.size):
            self.lastIndex = 0
            self.full = True
        else:
            self.dataset['state'][self.lastIndex,...] = state
            self.dataset['action'][self.lastIndex,...] = action
            self.dataset['reward'][self.lastIndex,...] = reward
            self.dataset['nextState'][self.lastIndex,...] = nextState
            self.dataset['terminated'][self.lastIndex,...] = terminated

            self.lastIndex += 1
            
        
    def sample(self,batchSize):
        randomIndex = np.random.choice(self.size, batchSize, replace=False)
        sampledBatch = {key: self.dataset[key][randomIndex]
                        for key in self.dataset.keys()
                        }
        return sampledBatch
    