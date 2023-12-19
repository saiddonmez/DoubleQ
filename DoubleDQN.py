import numpy as np
import torch
import game
from model import Model
from replay import replayBuffer
import torch.nn as nn
import torch.optim as optim
from time import perf_counter
from game import Game
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

# If cuda is available, use gpu
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

GAME = 'AssaultDeterministic-v4'
NUMACTIONS = 7


class DDQN():
    
    def __init__(self,GAME,NUMACTIONS,initEpsilon=1,finalEpsilon=0.1, epsilonDecreaseTime=100000,\
                 bufferSize=100000,batchSize=32,totalTrainSteps=5000000,lr=0.00025, discount=0.99,tau=10000,\
                 evaluationSteps=4500,evaluationEpisodes=10,evaluationFreq=100000, evaluationEpsilon = 0.05, noopActions = 8,\
                 momentum=0.95,render=False,modelPath=None,savedStatsPath=None,mode="DoubleDQN"):
        
        self.numactions = NUMACTIONS
        self.initEpsilon = initEpsilon
        self.finalEpsilon = finalEpsilon
        self.epsilon = self.initEpsilon
        self.epsilonDecreaseTime = epsilonDecreaseTime
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
        self.mode = mode

        #initialize models
        self.model = Model(self.numactions).to(device)
        self.targetModel = Model(self.numactions).to(device)

        self.evalStats = {"meanScores":[], "stdScores":[],"medianValues":[],"lowerValues":[],"higherValues":[],"meanLosses":[]}

        if modelPath != None:
            #to continue the training if it stopped for some reason
            self.model.load_state_dict(torch.load(modelPath))
            self.initEpsilon = self.finalEpsilon #to continue the training if it stopped for some reason
            with open(savedStatsPath,"rb") as f:
                self.evalStats = pickle.load(f)
            
            self.startingStep = int(modelPath.split('.')[0].split('_')[-1]) + 1 

        else:
            #train from scratch
            self.startingStep = 1

        self.targetModel.load_state_dict(self.model.state_dict())

        self.lossFunction = nn.SmoothL1Loss(reduction='mean')
        #self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr,momentum=momentum)
        self.optimizer = optim.Adam(self.model.parameters(),lr=lr)

        #initialize replay buffer
        self.replayBuffer = replayBuffer(self.bufferSize)

        #initialize environment
        if render:
            self.worker = Game(GAME, renderMode = "human", seed = 47)
        else:
            self.worker = game.Worker(GAME,None,8)
            self.worker.child.send(('reset',None))
            self.state = self.worker.child.recv() # Stacked 4 consecutive frames of the game

    def takeAction(self,epsilon,evaluate=False,action=None):


        state = torch.tensor(self.state,dtype=torch.float32,device=device) / 255.0
        actionSelected, value = self.actionSample(state,epsilon)

        if action == None:
            action = actionSelected

        self.worker.child.send(('step',action))
        nextState, reward, done, info = self.worker.child.recv()

        if not evaluate:
            self.replayBuffer.recordSample(self.state,action,reward,nextState,done)
            
        if done:
            self.worker.child.send(('reset',None))
            self.state = self.worker.child.recv() # Stacked 4 consecutive frames of the game
        else:
            self.state = nextState

        return reward, done, value
    
    def takeAction2(self,epsilon,evaluate=False,action=None):

        state = torch.tensor(self.state,dtype=torch.float32,device=device) / 255.0
        actionSelected, value = self.actionSample(state,epsilon)
        if action == None:
            action = actionSelected

        nextState, reward, done, info = self.worker.step(action)

        if not evaluate:
            self.replayBuffer.recordSample(self.state,action,reward,nextState,done)
            
        if done:
            self.state = self.worker.reset() # Stacked 4 consecutive frames of the game
        else:
            self.state = nextState

        return reward, done, value        

    def train(self):
        step = self.startingStep

        t0 = perf_counter()
        totalTime = [0,0]
        losses = 0

        
        
        for e in range(self.totalTrainSteps):

            t1 = perf_counter()
            self.takeAction(self.epsilon)

            if self.replayBuffer.full:
                t2 = perf_counter()

                totalTime[0] += t2-t1

                #Linearly decreasing epsilon during first 1M frames from 1 to 0.1 then fixed after 1M frames.
                if step < self.epsilonDecreaseTime:
                    self.epsilon = self.initEpsilon - (self.initEpsilon-self.finalEpsilon)*step/self.epsilonDecreaseTime
                else:
                    self.epsilon = self.finalEpsilon
            

                batch = self.replayBuffer.sample(self.batchSize)

                state = torch.tensor(batch['state'],dtype=torch.float32,device=device) / 255.0
                action = torch.from_numpy(batch['action']).to(device)
                nextState = torch.tensor(batch['nextState'],dtype=torch.float32,device=device) / 255.0
                reward = torch.from_numpy(batch['reward']).to(device)
                terminated = torch.from_numpy(batch['terminated'].astype(int)).to(device)
                
                self.optimizer.zero_grad()
                q = self.model(state)
                target = self.calculateTarget(nextState,reward,terminated)

                q_a = q.gather(-1,action.unsqueeze(-1)).squeeze(-1)
                loss = self.lossFunction(q_a,target)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

                self.optimizer.step()
           
                step += 1

                t3 = perf_counter()

                totalTime[1] += t3-t2
                losses += loss.detach()

            if step % self.evaluationFreq == 0:

                print(f"total loss in the last {self.evaluationFreq} steps is {losses}")

                meanTotalReward, stdTotalReward, medianTotalValue, lowerPercentile, higherPercentile = self.evaluate()

                now = datetime.now()

                self.evalStats['meanScores'].append(meanTotalReward)
                self.evalStats['stdScores'].append(stdTotalReward)
                self.evalStats['medianValues'].append(medianTotalValue)
                self.evalStats['lowerValues'].append(lowerPercentile)
                self.evalStats['higherValues'].append(higherPercentile)
                self.evalStats['meanLosses'].append(losses/self.evaluationFreq)

                losses=0

                print(f"{step} step , total time: {perf_counter()-t0}")

                print(f"total time for simulation in {step} steps is {totalTime[0]}")
                print(f"total time for training in {step} steps is {totalTime[1]}")

                torch.save(self.model.state_dict(),f"./savedModels/{self.mode}_{GAME.split('/')[-1]}_step_{step}.pth")
                torch.save(self.bestWeights,f"./bestSavedModels/{self.mode}_{GAME.split('/')[-1]}_step_{step}.pth")

                with open(f"{self.mode}_evalStats_"+now.strftime("%Y_%m_%d_%I_%M%p")+".pkl","wb") as f:
                    pickle.dump(self.evalStats, f)

            if step % self.tau == 0:

                self.targetModel.load_state_dict(self.model.state_dict())

        self.worker.child.send(("close", None))

    def actionSample(self,state,epsilon):
        with torch.no_grad():
            q = self.model(state)
            if np.random.rand()<epsilon:
                return np.random.choice(self.numactions), torch.max(q).cpu()
            else:
                return torch.argmax(q).cpu(), torch.max(q).cpu()

    def calculateTarget(self,nextState,reward,terminated):

        with torch.no_grad():
            q_tilda = self.targetModel(nextState)
            if self.mode == "DoubleDQN":
                q = self.model(nextState)
                bestAction = torch.argmax(q,1)
                target = reward + (1-terminated) * self.discount*q_tilda.gather(-1, bestAction.unsqueeze(-1)).squeeze(-1) #double q idea
            if self.mode == "DQN":
                target = reward + (1-terminated) * self.discount * torch.max(q_tilda,1).values
        return target
        
    def evaluate(self):


        TotalRewards = []
        totalValues = []
        count = 0

        for agentNo in range(self.evaluationEpisodes):
            #initialize the game
            self.worker.child.send(('reset',None))
            self.state = self.worker.child.recv() # Stacked 4 consecutive frames of the game
            totalReward = 0
            totalValue = 0
            for k in range(self.noopActions): #number of no operation actions
                reward, done, value = self.takeAction(self.evaluationEpsilon,evaluate=True,action=0)
                totalReward += reward
                totalValue += value
                count+=1
            
            for j in range(self.evaluationSteps-self.noopActions):
                reward, done, value = self.takeAction(self.evaluationEpsilon,evaluate=True)
                totalReward += reward
                totalValue += value
                count+=1

                if done:
                    break

            totalValues.append(totalValue.detach().numpy())
            TotalRewards.append(totalReward)

            medianTotalValue = np.median(totalValues)

            lowerPercentile = np.percentile(totalValues,0.1)
            higherPercentile = np.percentile(totalValues,0.9)

            meanTotalReward = np.mean(TotalRewards)
            stdTotalReward = np.std(TotalRewards)

        if meanTotalReward > self.bestTotalReward:
            self.bestTotalReward = meanTotalReward
            self.bestWeights = self.model.state_dict()

        return meanTotalReward, stdTotalReward, medianTotalValue, lowerPercentile, higherPercentile
        
    def render(self):

        self.state = self.worker.reset()
        totalReward = 0

        for k in range(self.noopActions): #number of no operation actions
            reward,done,value = self.takeAction2(0.002,evaluate=True,action=0)
            totalReward += reward
            self.worker.env.render()
        
        for j in range(self.evaluationSteps):
            reward,done,value = self.takeAction2(0.002,evaluate=True)
            totalReward += reward
            self.worker.env.render()
        
        self.worker.child.send(("close", None))
            
if __name__ == "__main__":

    mode = "DQN"
    #modelPath = r"savedModels\AssaultDeterministic-v4_step_2400000.pth" #to continue training or to render performance
    modelPath = r"savedModels\DQN_AssaultDeterministic-v4_step_4900000.pth"
    #modelPath = r"savedModels\AssaultDeterministic-v4_step_5000000.pth"
    savedStatsPath = r"DQN_evalStats_2023_12_19_04_29AM.pkl"

    #modelPath = None  #to train from scratch
    #savedStatsPath = None #to train from scratch

    agent = DDQN(GAME,NUMACTIONS,modelPath=modelPath,savedStatsPath=savedStatsPath,mode=mode)
    agent.train()


    # agent = DDQN(GAME,NUMACTIONS,modelPath = modelPath, render=True, savedStatsPath=savedStatsPath)
    # agent.render()


#%%
# import matplotlib.pyplot as plt
# import numpy as np
# import pickle
# with open(f"DQN_evalStats_2023_12_19_04_29AM.pkl","rb") as f:
#     evalStatsDQN = pickle.load(f)

# with open(f"evalStats_2023_12_18_09_01AM.pkl", "rb") as f:
#     evalStatsDDQN = pickle.load(f)

# plt.plot(evalStatsDQN['medianValues'])
# plt.plot(evalStatsDDQN['medianValues'])
# %%
