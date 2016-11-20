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
ALPHA = 0.5
GAMMA = 0.7
# NO_OF_EPISODES = 500
NO_OF_EPISODES = 200
NO_OF_ITER = 5000
# TILE_SIZE = 0.2
# VELOCITY_TILE_SIZE = 0.2
# ACTION_TILE_SIZE = 1
ANGLE_TILE_SIZE = 1
env = gym.make('Pendulum-v0')
# ACTIONS = [-2, -1, -0.5, 0, 0.5, 1, 2]
ACTIONS = [-2, 0, 2]
VELOCITY = [x for x in range(-8, 9)]
MAX_LENGTH_MEMORY = 50

def coordinate_check():
    first_obs = env.reset()
    print "First OBS %s " %first_obs
    print "Constructed degrees %s" % (cart2pol(first_obs[0], first_obs[1]))
    return

# In[2]:

def epsilon_greedy((theta,thetadot), epsilon):
    state = (theta, thetadot)
    # print state
    valid_actions = Q[state].keys()
    # max_action = max(Q[state], key=Q[state].get)
    if np.random.random() < epsilon:
#         print 'random action'
        random_action = ACTIONS[np.random.randint(len(ACTIONS))]
        return random_action
    # print state
    max_action = max(Q[state], key=Q[state].get)
    # if theta == 90 :
    #     print 'Selected %s action for %s' % (max_action, Q[state])
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
    # theta = get_bin(cart2pol(S[0], S[1]), 0, ANGLE_TILE_SIZE)
    theta = int(cart2pol(S[0], S[1]))
    # thetadot = get_velocity_tile(S[2])
    # thetadot = math.floor(S[2])
    thetadot = int(S[2])
    return (theta, thetadot)

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
#     return(rho, phi)
    # print "Angle reconstucted %s"%phi
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
        for thetadot in VELOCITY:
            for action in ACTIONS:
               # Q[(theta, thetadot)][action] = -20
               Q[(theta, thetadot)][action] = 0


# In[6]:

def qlearning():
    init_Q()
    episode_rewards = []
    episode_steps = []
    replay_memory = []
    epsilon = 0.1
    epsilon_d = 0.99
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
            # Get the action for this state
            A = epsilon_greedy(discrete_S, epsilon) 
            
            # print "State:%s, Action:%s"%(discrete_S, ((A*ACTION_TILE_SIZE) - 2))
            # Execute the step
            S_next, reward, _, _ = env.step([A])
            # print S_next
            total_episode_reward += reward
            
            discrete_S_next = get_discrete_state(S_next)
            
            # Add observation to replay memory
            replay_memory.append([discrete_S, A, reward, discrete_S_next])

            A_next = max(Q[discrete_S_next], key=Q[discrete_S_next].get)
            Q[discrete_S][A] = Q[discrete_S][A] + ALPHA*(reward + GAMMA*Q[discrete_S_next][A_next] - Q[discrete_S][A])
            S = S_next
            no_of_steps_per_episode += 1

            # Experience Replay
            for item in replay_memory:
                replay_A_next = max(Q[item[3]], key=Q[item[3]].get)
                Q[item[0]][item[1]] = Q[item[0]][item[1]] + ALPHA * (item[2] + GAMMA*Q[item[3]][replay_A_next] - Q[item[0]][item[1]])
                if len(replay_memory) > MAX_LENGTH_MEMORY:
                    replay_memory.pop(0)

        # Update the rewards and the no of steps taken
        episode_rewards.append(total_episode_reward)
        episode_steps.append(no_of_steps_per_episode)

        # Epsilon Decay per episode
        epsilon = epsilon/epsilon_d
    return episode_rewards, episode_steps


# In[ ]:

episode_rewards, _ = qlearning()
print episode_rewards
# coordinate_check()
# get_position_tile(0, 0)


# In[ ]:

# plt.plot(episode_rewards)


# In[ ]:



