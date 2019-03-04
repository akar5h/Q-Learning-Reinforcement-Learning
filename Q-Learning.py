#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from random import randint, random


class FrozenLake:
    def __init__(self, N1, M1, alpha = 0.8, lambd = 0.9):
        # M less than N
        # 0 for left, 1 for right, 2 for top, 3 for bottom
        self.N = N1
        self.M = M1
        self.alpha = alpha
        self.lambd = lambd
        
        self.max_epis = 10000
        self.epsilon = 0.5 
        self.epsilonDecay = 0.90
        self.max_steps = 1000
        
        self.Lake = []
        self.Q = []
        
        for i in range(self.N):
            self.Lake.append([0]*self.N)
            self.Q.append([])
            for j in range(self.N):
                self.Q[i].append([0]*4)
        
        no_holes = 0
        while no_holes<self.M:
            i = randint(0, self.N - 1)
            j = randint(0, self.N - 1)
            if self.Lake[i][j]==0:
                self.Lake[i][j] = -3
                no_holes += 1
        
        self.start = (0, 0)
        self.goal = (self.N - 1, self.N - 1)
        self.Lake[self.N -1][self.N - 1] = 9
    
    def next_state(self, state, action):
        i = state[0]
        j = state[1]
        #print(next_s[1])         
        if action==0:
            next_s = [i-1, j]
        elif action==1:
            next_s = [i+1, j]
        elif action==2:
            next_s = [i, j-1]
        elif action==3:
            next_s = [i, j+1] 
        
        if next_s[0]<0:
            return -1
        if next_s[0]>(self.N-1):
            return -1
        if next_s[1]<0: 
            return -1
        if next_s[1]>(self.N-1):
            return -1
        return next_s
    
    def select_action(self, state):
        i = state[0]
        j = state[1]
        if random()<self.epsilon:
            while True:
                action = randint(0, 3)
                if self.next_state(state, action)!=-1:
                    return action
        else:
            max_Q = min(self.Q[i][j])
            for k in range(0, 4):
                if self.next_state(state, k)!=-1:
                    if self.Q[i][j][k] >= max_Q:
                        max_Q = self.Q[i][j][k]
                        action = k
            self.epsilon = self.epsilon*self.epsilonDecay
        return action
    
    def update_Q(self, curr_state, next_, action, reward):
        i = curr_state[0]; j = curr_state[1]
        max_Q = min(self.Q[next_[0]][next_[1]])
        for k in range(0, 4):
            if self.next_state(curr_state, k)!=-1:
                if self.Q[next_[0]][next_[1]][k] >= max_Q:
                    max_Q = self.Q[next_[0]][next_[1]][k]
        self.Q[i][j][action] = (1 - self.alpha)*self.Q[i][j][action] + self.alpha*(reward + (self.lambd*max_Q))
    
    def learn_Q(self):
        reward_lis = []
        for epis in range(self.max_epis):
            epis_reward = 0
            curr_state = [randint(0, self.N -1), randint(0, self.N - 1)]
            count = 1000
            while True:
                action = self.select_action(curr_state)
                next_ = self.next_state(curr_state, action)
                reward = self.Lake[next_[0]][next_[1]]
                self.update_Q(curr_state, next_, action, reward)
                curr_state = next_
                epis_reward += reward
                count += 1
                if tuple(curr_state)==self.goal:
                    break
                if count>1000:
                    break
            
            reward_lis.append(epis_reward)
        return reward_lis
    
    def printLake(self):
        for i in range(self.N):
            for j in range(self.N):
                print(self.Lake[i][j], end = " ")
            print()
    
    def printQ(self):
        for i in range(self.N):
            for j in range(self.N):
                print(max(self.Q[i][j]), end = " ")
            print()
    




Lake = FrozenLake(10, 12, 0.3, 0.4)
lis = Lake.learn_Q()
plt.scatter(range(10000), lis)
plt.title("N = 10, M = 12, alpha = 0.3, lamda = 0.4")
plt.xlabel("No of Episodes")
plt.ylabel("Reward")
plt.show()




Lake = FrozenLake(10, 12, 0.5, 0.4)
lis = Lake.learn_Q()
plt.scatter(range(10000), lis)
plt.title("N = 10, M = 12, alpha = 0.5, lamda = 0.4")
plt.xlabel("No of Episodes")
plt.ylabel("Reward")
plt.show()




Lake = FrozenLake(10, 12, 0.7, 0.4)
lis = Lake.learn_Q()
plt.scatter(range(10000), lis)
plt.title("N = 10, M = 12, alpha = 0.7, lamda = 0.4")
plt.xlabel("No of Episodes")
plt.ylabel("Reward")
plt.show()



Lake = FrozenLake(10, 12, 0.9, 0.4)
lis = Lake.learn_Q()
plt.scatter(range(10000), lis)
plt.title("N = 10, M = 12, alpha = 0.9, lamda = 0.4")
plt.xlabel("No of Episodes")
plt.ylabel("Reward")
plt.show()




Lake = FrozenLake(10, 12, 0.5, 0.2)
lis = Lake.learn_Q()
plt.scatter(range(10000), lis)
plt.title("N = 10, M = 12, alpha = 0.5, lamda = 0.2")
plt.xlabel("No of Episodes")
plt.ylabel("Reward")
plt.show()



