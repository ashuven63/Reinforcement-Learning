# coding: utf-8

# In[1]:

import gym
import numpy as np
from collections import defaultdict 
import random
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import matplotlib
import math
import sys, traceback

Q = defaultdict(dict)
# EPSILON = 0.2
# EPSILON_DECAY = 0.98
ALPHA = 0.1
GAMMA = 0.9
# NO_OF_EPISODES = 500
NO_OF_EPISODES = 3500
NO_OF_ITER = 1000
# TILE_SIZE = 0.2
VELOCITY_TILE_SIZE = 0.2
ACTION_TILE_SIZE = 1
ANGLE_TILE_SIZE = 1
env = gym.make('Pendulum-v0')

# In[2]:

def epsilon_greedy((theta,thetadot), epsilon):
    state = (theta, thetadot)
    # print state
    valid_actions = Q[state].keys()
    # max_action = max(Q[state], key=Q[state].get)
    if np.random.random() < epsilon:
#         print 'random action'
        random_action = np.random.randint(int(4/ACTION_TILE_SIZE))
        return random_action
    max_action = max(Q[state], key=Q[state].get)
    if theta == 90 :
        print 'Selected %s action for bin %s for %s' % (((max_action*ACTION_TILE_SIZE) - 2), max_action, Q[state])
    return max_action


# In[3]:

def get_bin(coor, offset, t):
    quo = int((coor+offset) / t)
    # print quo
    mod = (coor+offset) % t
    # print mod
    if quo == 0:
        binno = 0
    elif mod == 0:
        binno = quo - 1
    elif quo == int((offset*2) / t):
        binno = quo - 1
    else:
        binno = quo
    return binno
# print get_bin(-1,1, TILE_SIZE)
# print get_bin(-0.344,1, TILE_SIZE)
# print get_bin(0.0, 1, TILE_SIZE)
# print get_bin(0.02,1, TILE_SIZE)
# print get_bin(0.,1, TILE_SIZE)
# print get_bin(0.356,1, TILE_SIZE)
# print get_bin(1.0,1, TILE_SIZE)
# print get_bin(1.,1, TILE_SIZE)
        
# def get_position_tile(i, j):
#     x_coor = get_bin(i, 1, TILE_SIZE)
#     y_coor = get_bin(j, 1, TILE_SIZE)
#     return (x_coor, y_coor)
    
def get_velocity_tile(i):
    ang_vel = get_bin(i, 8, VELOCITY_TILE_SIZE)
    return ang_vel

def get_discrete_state(S):
    # (x_coor, y_coor) = get_position_tile(S[0], S[1])
    theta = get_bin(S[0], 180, ANGLE_TILE_SIZE)
    thetadot = get_velocity_tile(S[1])
    return (theta, thetadot)

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
#     return(rho, phi)
    angle = math.degrees(phi)
    if angle < 0:
        angle = 360 + angle
    return angle

# def get_discrete_action(A):
# #     return int((A+2)/ACTION_TILE_SIZE)
#     return get_bin(A, 2, ACTION_TILE_SIZE)
#     print(env.action_space.high)
#     print(env.action_space.low)

def init_Q():
    for theta in range(0,int(360/ANGLE_TILE_SIZE)):
        for thetadot in range(0, int(16/VELOCITY_TILE_SIZE)):
            for action in range(int(4/ACTION_TILE_SIZE)):
               # Q[(theta, thetadot)][action] = -20
               Q[(theta, thetadot)][action] = 0


# In[6]:

def qlearning():
    init_Q()
    episode_rewards = []
    episode_steps = []
    epsilon = 0.1
    epsilon_d = 1
    for i in range(NO_OF_EPISODES):
        print('Episode Start %s'%(i))
        S = env.reset()
        # print 'First Observation %s'%(S)
        total_episode_reward = 0
        no_of_steps_per_episode = 0
        done = False # When should this end because this is not a episodic task. 
        for t in range(NO_OF_ITER): # change the terminal state 
            
            if i == NO_OF_EPISODES -1 :
                env.render()
            discrete_S = get_discrete_state(S)
            A = epsilon_greedy(discrete_S, epsilon) # Should change the state here
            
            # print "State:%s, Action:%s"%(discrete_S, ((A*ACTION_TILE_SIZE) - 2))
            S_next, reward, _, _ = env.step([((A*ACTION_TILE_SIZE) - 2)])
            # print S_next
            total_episode_reward += reward
            
            discrete_S_next = get_discrete_state(S_next)
            # if discrete_S_next[0] == 90 :
            #     print 'Bonus for %s'%(S_next)
            #     bonus = 10
            # else:
            #     bonus = 0
            try:
                A_next = max(Q[discrete_S_next], key=Q[discrete_S_next].get)
            except:
                # print Q
                print S_next
                get_discrete_state(S_next)
                print discrete_S, A, discrete_S_next, A_next
                traceback.print_exc(file=sys.stdout)
                return
            Q[discrete_S][A] = Q[discrete_S][A] + ALPHA*(reward + GAMMA*Q[discrete_S_next][A_next] - Q[discrete_S][A])
            S = S_next
            no_of_steps_per_episode += 1
        episode_rewards.append(total_episode_reward)
        episode_steps.append(no_of_steps_per_episode)
        epsilon = epsilon/epsilon_d
    # for key in Q:
    #     if key[0] == 90:
    #         print key
    #         for k in Q[key]:
    #             # if Q[key][k] != 0:
    #             print "Degree:%s, Ang_vel:%s Val:%s"%(key, k, Q[key][k]) 
    return episode_rewards, episode_steps


# In[ ]:

episode_rewards, _ = qlearning()
print episode_rewards

# get_position_tile(0, 0)


# In[ ]:

# plt.plot(episode_rewards)


# In[ ]:



