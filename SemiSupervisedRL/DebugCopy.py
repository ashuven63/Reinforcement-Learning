
# coding: utf-8

# In[105]:

# get_ipython().magic(u'matplotlib inline')
import gym
import numpy as np
from collections import defaultdict 
import random
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import matplotlib
import math
import sys, traceback
import logging
import pickle

# In[110]:

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()
Q = defaultdict(dict)
Observed_Counts = defaultdict(dict)
ALPHA = 0.1
GAMMA = 0.9
NO_OF_EPISODES = 1000
NO_OF_ITER = 1000
ANGLE_TILE_SIZE = 18 # Should be such that no bin lies on both sides of center
THETADOT_TILE_SIZE = 6
env = gym.make('Pendulum-v0')
ACTIONS = [-2, -1.6, -0.8, -0.4, -0.2, 0, 0.2, 0.4, 0.8, 1.6, 2]
VELOCITY = [x for x in range(-8, 9)]
MAX_LENGTH_MEMORY = 10000
REPLAY_SAMPLE_SIZE = 5000
ENABLE_REPLAY = True
RENDER = True
DEBUG = True
replay_memory = []


# In[111]:

thetadot_bin = {}
def epsilon_greedy(state, epsilon):
    # print state
    valid_actions = Q[state].keys()
    if np.random.random() < epsilon:
        return np.random.choice(ACTIONS)
    max_action = max(Q[state], key=Q[state].get)
    return max_action

def set_thetadot_bins():
    thetadot_bin[1] = (0, 0.5)
    thetadot_bin[-1] = (-0.5, 0)
    thetadot_bin[2] = (0.5, 1.5)
    thetadot_bin[3] = (1.5, 2.5)
    thetadot_bin[4] = (2.5, 3.5)
    thetadot_bin[5] = (3.5, 4.5)
    thetadot_bin[6] = (4.5, 5.5)
    thetadot_bin[7] = (5.5, 6.5)
    thetadot_bin[8] = (6.5, 8.1)
    
    thetadot_bin[-2] = (-1.5, -0.5)
    thetadot_bin[-3] = (-2.5, -1.5)
    thetadot_bin[-4] = (-3.5, -2.5)
    thetadot_bin[-5] = (-4.5, -3.5)
    thetadot_bin[-6] = (-5.5, -4.5)
    thetadot_bin[-7] = (-6.5, -5.5)
    thetadot_bin[-8] = (-8.1, -6.5)

set_thetadot_bins()

def get_theta_bin(angle):
    bin_num = int(angle/ANGLE_TILE_SIZE)
    return bin_num
    
def get_thetadot_bin(i):
    for key in thetadot_bin:
        if i > thetadot_bin[key][0] and i <= thetadot_bin[key][1]:
            return key
    raise "Key not found for %s"%(i)

# print get_thetadot_bin(8)
# print get_thetadot_bin(-8)
# print get_thetadot_bin(-7.5)
# print get_thetadot_bin(7.5)
# print get_thetadot_bin(0)
# print get_thetadot_bin(0.2)
# print get_thetadot_bin(-0.5)
# print get_thetadot_bin(-0.533)
# print get_thetadot_bin(-0.4)
    
def get_discrete_state(S):
    theta = get_theta_bin((math.degrees(normalize_angle(S[0])))) 
    thetadot = get_thetadot_bin(S[1])
    return (theta, thetadot)

def normalize_angle(x):
    x = (x % (2*np.pi))
    if x < 0:
        return ( x + (2*np.pi))
    return x

def init_Q():
    angle_incr_step = (1.8/(180/ANGLE_TILE_SIZE))
    for theta in range(0, 360/ANGLE_TILE_SIZE):
        for thetadot in thetadot_bin.keys():
            for action in ACTIONS:
                if theta > 180:
                    Q[(theta, thetadot)][action] = 0.2 + (angle_incr_step * (360 - theta))
                else:
                    Q[(theta, thetadot)][action] = -(0.2 + (angle_incr_step * (theta)))
                Observed_Counts[(theta, thetadot, action)] = 0

def replay():
    if ENABLE_REPLAY:
        # Add observation to replay memory
        choices = np.array(replay_memory)
        idx = np.random.choice(len(choices), REPLAY_SAMPLE_SIZE)
        current_sample_set = choices[idx]
        for item in current_sample_set:
            replay_A_next = max(Q[item[3]], key=Q[item[3]].get)
            Q[item[0]][item[1]] = Q[item[0]][item[1]] + ALPHA * (item[2] + GAMMA*Q[item[3]][replay_A_next] - Q[item[0]][item[1]])


# In[112]:

def plot_hist(v, b, name):
    plt.figure()
    plt.hist(v, bins=b)
    # plt.show()
    plt.savefig(name)
    plt.close()
    
def plot_episode_rewards(episode_rewards):
    plt.figure()
    plt.plot(range(NO_OF_EPISODES), episode_rewards)
    plt.savefig("EpisodeRewards")
    plt.close()


# In[113]:

def qlearning():
    init_Q()
    episode_rewards = []
    observed_theta, observed_thetadot = [], []
    all_observed_theta, all_observed_thetadot = [], []
    epsilon = 0.1
    epsilon_d = 1
    for i in range(NO_OF_EPISODES):
        print 'Episode {0}'.format(i)
        if i % 100 == 0:
#             logger.debug('Episode {0}'.format(i))
            if DEBUG:
                fn = "%d"%(i)
                plot_hist(observed_thetadot, range(-8, 9), fn+"thetadot")
                plot_hist(observed_theta, 360, fn+"theta")
                observed_theta = []
                observed_thetadot = []
        S = env.reset()
        discrete_S = get_discrete_state(S)
        total_episode_reward = 0
        for t in range(NO_OF_ITER):
            if RENDER:
                if i == NO_OF_EPISODES -1 :
                    env.render() 
            A = epsilon_greedy(discrete_S, epsilon)
            # if DEBUG:
            #     if i > NO_OF_EPISODES - 6:
            #         print "Selected %s Action for State %s with Value %s"%(A, discrete_S, Q[discrete_S][A])
            # Execute the step
            S_next, reward, _, _ = env.step([A])
            total_episode_reward += reward
            # If debugging observe the counts of each state. 
            if DEBUG:
                Observed_Counts[(discrete_S[0], discrete_S[1], A)] += 1
                observed_theta.append(discrete_S[0])
                observed_thetadot.append(S[1])
                all_observed_theta.append(discrete_S[0])
                all_observed_thetadot.append(S[1])
            discrete_S_next = get_discrete_state(S_next)
            A_next = max(Q[discrete_S_next], key=Q[discrete_S_next].get)
            # Update the Q values
            Q[discrete_S][A] = Q[discrete_S][A] + ALPHA*(reward + GAMMA*Q[discrete_S_next][A_next] - Q[discrete_S][A])
            discrete_S = discrete_S_next
            # Experience Replay
            if len(replay_memory) == MAX_LENGTH_MEMORY:
                replay_memory.pop(0)
            replay_memory.append([discrete_S, A, reward, discrete_S_next])
            replay()            

        # if DEBUG:
        #     if i > NO_OF_EPISODES - 6:
        #         print "************************************************"
        # Update the rewards and the no of steps taken
        episode_rewards.append(total_episode_reward)
        # Epsilon Decay per episode
        epsilon = epsilon/epsilon_d
        pickle.dump(Q, open("Q.p", "wb"))
        pickle.dump(Observed_Counts, open("ObservedCounts.p", "wb"))
    return episode_rewards


# In[ ]:

episode_rewards = qlearning()
plot_episode_rewards(episode_rewards)


# In[102]:

plot_episode_rewards(episode_rewards)


# In[ ]:

if DEBUG:
    # for key in Observed_Counts:
    #     print key
    #     print Observed_Counts[key]
    #     print '***********************************************'
    # print '############################################'
    # for key in 
    # plt.hist(observed_thetadot, bins=range(-8, 9))
    # plt.show()
    # plt.hist(observed_theta, bins=360)
    # plt.show()
    for key in Q:
        if (key[0] >= 0 and key[0] <=10) or (key[0] >=350):
            print key
            print Q[key]
    print "********************************************************************************************************************************************"
    for key in Observed_Counts:
        if (key[0] >= 0 and key[0] <=10) or (key[0] >=350):
            print "Key: %s, Value:%s" % (key, Observed_Counts[key])
    # fn = "all"
#     plot_hist(all_observed_thetadot, range(-8, 9), fn+"thetadot")
#     plot_hist(all_observed_theta, 360, fn+"theta")
    # pass


# In[ ]:



