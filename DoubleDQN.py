import numpy as np
import torch
import game
from model import Model
from replay import replayBuffer
import torch.nn as nn
import torch.optim as optim

# If cuda is available, use gpu
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

GAME = 'BreakoutNoFrameskip-v4'
NUMACTIONS = 4

# class lossFunc(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, predictions, targets):
#         return torch.mean((predictions - targets) ** 2)

class DDQN():
    
    def __init__(self,GAME,NUMACTIONS,initEpsilon=1,finalEpsilon=0.1, epsilonResetTime=1e6,\
                 bufferSize=1e6,batchSize=32,totalTrainSteps=5e7,lr=0.00025, discount=0.99,tau=10000,\
                 evaluationSteps=4500,evaluationEpisodes=100,evaluationFreq=1e6, evaluationEpsilon = 0.05, noopActions = 30,\
                 momentum=0.95):
        
        self.numactions = NUMACTIONS
        self.initEpsilon = initEpsilon
        self.finalEpsilon = finalEpsilon
        self.epsilon = self.initEpsilon
        self.epsilonResetTime = epsilonResetTime
        self.bufferSize = bufferSize
        self.batchSize = batchSize
        self.totalTrainSteps = totalTrainSteps
        self.discount = discount
        self.evaluationSteps = evaluationSteps
        self.evaluationEpisodes = evaluationEpisodes
        self.evaluationFreq = evaluationFreq
        self.evaluationEpsilon = evaluationEpsilon
        self.noopActions = noopActions
        self.tau = tau
        self.bestTotalReward = -np.Inf
        self.bestWeights = None

        #initialize models
        self.model = Model(self.numactions).to(device)
        self.targetModel = Model(self.numactions).to(device)
        self.targetModel.load_state_dict(self.model.state_dict())

        self.lossFunction = torch.nn.HuberLoss(reduction='mean', delta=1)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr,momentum=momentum)

        #initialize replay buffer
        self.replayBuffer = replayBuffer(self.bufferSize)

        #initialize environment
        self.worker = game.Worker(GAME,47)

        self.worker.child.send('reset')
        self.state = self.worker.child.recv() # Stacked 4 consecutive frames of the game

    def takeAction(self,epsilon,evaluate=False,action=None):

        if action == None:
            action = self.actionSample(self.state,self.epsilon)

        self.worker.child.send(('step',action))
        nextState, reward, done, info = self.worker.child.recv()

        if not evaluate:
            self.replayBuffer.recordSample(self.state,action,reward,nextState,done)
            
        if done:
            self.worker.child.send('reset')
            self.state = self.worker.child.recv() # Stacked 4 consecutive frames of the game
        else:
            self.state = nextState

        return reward

    def train(self):
        step = 0
        
        for e in range(self.totalTrainSteps):


            self.takeAction(self.epsilon)

            if self.replayBuffer.full:

                #Linearly decreasing epsilon during first 1M frames from 1 to 0.1 then fixed after 1M frames.
                if step < self.epsilonResetTime:
                    self.epsilon = self.initEpsilon - (self.initEpsilon-self.finalEpsilon)*step/self.epsilonResetTime
                else:
                    self.epsilon = self.finalEpsilon
            

                batch = self.replayBuffer.sample(self.batchSize)

                state = torch.from_numpy(batch['state']).to(device)
                action = torch.from_numpy(batch['action']).to(device)
                nextState = torch.from_numpy(batch['nextState']).to(device)
                reward = torch.from_numpy(batch['reward']).to(device)
                terminated = torch.from_numpy(batch['terminated']).to(device)
                
                q = self.model(state)
                target = self.calculateTarget(nextState,reward,terminated)

                loss = self.loss(q,target)
                loss.backward()
           
                step += 1

            if step % self.evaluationFreq == 0:

                meanTotalReward, stdTotalReward = self.evaluate()

        pass

    def actionSample(self,state,epsilon):

        if np.random.rand()<epsilon:
            return np.random.choice(self.numactions)
        else:
            with torch.no_grad():
                q = self.model(state)
                return torch.argmax(q)

    def calculateTarget(self,nextState,reward,terminated):

        with torch.no_grad():

            if terminated:
                target = reward
            else:
                q = self.model(nextState)
                q_tilda = self.targetModel(nextState)
                target = reward + self.discount*q_tilda(torch.argmax(q)) #double q idea

        return target
        
    def evaluate(self):

        #initialize the game
        self.worker.child.send('reset')
        self.state = self.worker.child.recv() # Stacked 4 consecutive frames of the game
        TotalRewards = []
        for agentNo in range(self.evaluationAgents):
            totalReward = 0
            for k in range(self.noopActions): #number of no operation actions
                reward = self.takeAction(self.evaluationEpsilon,evaluate=True,action=0)
                totalReward += reward
            
            for j in range(self.evaluationSteps):
                self.takeAction(self.evaluationEpsilon,evaluate=True)
                totalReward += reward

            TotalRewards.append(totalReward)
            meanTotalReward = np.mean(TotalRewards)
            stdTotalReward = np.std(TotalRewards)

        if meanTotalReward > self.bestTotalReward:
            self.bestTotalReward = meanTotalReward
            self.bestWeights = self.model.state_dict()

        return meanTotalReward, stdTotalReward
        

            
            
