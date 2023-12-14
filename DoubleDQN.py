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

# If cuda is available, use gpu
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

GAME = 'AssaultNoFrameskip-v4'
NUMACTIONS = 7


class DDQN():
    
    def __init__(self,GAME,NUMACTIONS,initEpsilon=1,finalEpsilon=0.1, epsilonResetTime=1000,\
                 bufferSize=1000,batchSize=32,totalTrainSteps=500000,lr=0.0025, discount=0.99,tau=100,\
                 evaluationSteps=450,evaluationEpisodes=10,evaluationFreq=1000, evaluationEpsilon = 0.05, noopActions = 8,\
                 momentum=0.95,render=False,modelPath=None):
        
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

        if modelPath != None:
            self.model.load_state_dict(torch.load(modelPath))

        self.targetModel.load_state_dict(self.model.state_dict())

        self.lossFunction = torch.nn.HuberLoss(reduction='mean', delta=1)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr,momentum=momentum)

        #initialize replay buffer
        self.replayBuffer = replayBuffer(self.bufferSize)

        #initialize environment
        if render:
            self.worker = Game(GAME, renderMode = "human", seed = 47)
        else:
            self.worker = game.Worker(GAME,None,47)
            self.worker.child.send(('reset',None))
            self.state = self.worker.child.recv() # Stacked 4 consecutive frames of the game

    def takeAction(self,epsilon,evaluate=False,action=None):


        state = torch.tensor(self.state,dtype=torch.float32,device=device) / 255.0
        actionSelected, value = self.actionSample(state,self.epsilon)

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
        if action == None:
            state = torch.tensor(self.state,dtype=torch.float32,device=device) / 255.0
            action, value = self.actionSample(state,self.epsilon)

        
        nextState, reward, done, info = self.worker.step(action)

        if not evaluate:
            self.replayBuffer.recordSample(self.state,action,reward,nextState,done)
            
        if done:
            self.state = self.worker.reset() # Stacked 4 consecutive frames of the game
        else:
            self.state = nextState

        return reward, done, value        

    def train(self):
        step = 1

        t0 = perf_counter()
        totalTime = [0,0]

        evalStats = {"meanScores":[], "stdScores":[],"medianValues":[],"lowerValues":[],"higherValues":[]}
        
        for e in range(self.totalTrainSteps):

            t1 = perf_counter()
            self.takeAction(self.epsilon)

            if self.replayBuffer.full:
                t2 = perf_counter()

                totalTime[0] += t2-t1

                #Linearly decreasing epsilon during first 1M frames from 1 to 0.1 then fixed after 1M frames.
                if step < self.epsilonResetTime:
                    self.epsilon = self.initEpsilon - (self.initEpsilon-self.finalEpsilon)*step/self.epsilonResetTime
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

                self.optimizer.step()
           
                step += 1

                t3 = perf_counter()

                totalTime[1] += t3-t2

            if step % self.evaluationFreq == 0:

                meanTotalReward, stdTotalReward, medianTotalValue, lowerPercentile, higherPercentile = self.evaluate()

                evalStats['meanScores'].append(meanTotalReward)
                evalStats['stdScores'].append(stdTotalReward)
                evalStats['medianValues'].append(medianTotalValue)
                evalStats['lowerValues'].append(lowerPercentile)
                evalStats['higherValues'].append(higherPercentile)

                print(f"{step} step , total time: {perf_counter()-t0}")

                print(f"total time for simulation in {step} steps is {totalTime[0]}")
                print(f"total time for training in {step} steps is {totalTime[1]}")

                torch.save(self.model.state_dict(),f"{GAME}_step_{step}.pth")
                


            if step % self.tau == 0:

                self.targetModel.load_state_dict(self.model.state_dict())

        with open(f"evalStats.pkl","wb") as f:
            pickle.dump(evalStats, f)


    def actionSample(self,state,epsilon):
        q = self.model(state)
        if np.random.rand()<epsilon:
            return np.random.choice(self.numactions), torch.max(q).cpu()
        else:
            with torch.no_grad():          
                return torch.argmax(q).cpu(), torch.max(q).cpu()

    def calculateTarget(self,nextState,reward,terminated):

        with torch.no_grad():

            q = self.model(nextState)
            q_tilda = self.targetModel(nextState)
            bestAction = torch.argmax(q,1)
            target = reward + (1-terminated) * self.discount*q_tilda.gather(-1, bestAction.unsqueeze(-1)).squeeze(-1) #double q idea

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

        self.worker.reset()
        totalReward = 0

        for k in range(self.noopActions): #number of no operation actions
            reward,done,value = self.takeAction2(self.evaluationEpsilon,evaluate=True,action=0)
            totalReward += reward
            self.worker.env.render()
        
        for j in range(self.evaluationSteps):
            reward,done,value = self.takeAction2(self.evaluationEpsilon,evaluate=True)
            totalReward += reward
            self.worker.env.render()
        
            
if __name__ == "__main__":

    
    agent = DDQN(GAME,NUMACTIONS)
    agent.train()

    # modelPath = r"C:\Users\ahmet\Desktop\Bilkent_Grad\EEE548\Project\DoubleQ\AssaultNoFrameskip-v4_step_60000.pth"

    # agent = DDQN(GAME,NUMACTIONS,modelPath = modelPath, render=True)
    # agent.render()
