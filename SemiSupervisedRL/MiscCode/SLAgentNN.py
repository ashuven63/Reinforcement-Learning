
# coding: utf-8

# In[84]:

import numpy as np
import random
import csv
from reinforcementLearningCar.nn import neural_net, LossHistory
import os.path
import timeit
import gym
import matplotlib.pyplot as plt


# In[85]:

NUM_INPUT = 2
GAMMA = 0.9  # Forgetting.
TUNING = False  # If False, just use arbitrary, pre-selected params.
ACTIONS = [-2, -1.6, -0.8, -0.4, -0.2, 0, 0.2, 0.4, 0.8, 1.6, 2]
EPISODE_SIZE = 500
env = gym.make('Pendulum-v0')


# In[86]:

def train_net(model, params):
    filename = params_to_filename(params)
    observe = 1000  # Number of frames to observe before training.
    epsilon = 1
    t = 0
    episode_number, episode_reward = 0, 0
#     train_frames = 1000000  # Number of frames to obserce.
    train_frames = 10000  # Number of frames to obserce.
    batchSize = params['batchSize']
    buffer = params['buffer']
    print 'Executing %d episodes'%(int(train_frames/EPISODE_SIZE))
    replay = []  # stores tuples of (S, A, R, S').
    loss_log = []
    all_rewards = []
    
    state = env.reset()
    state_array = (np.array([state[0], state[1]])).reshape((1,2))
    while t < train_frames:
        t += 1

        # Choose an action.
        if random.random() < epsilon or t < observe:
            action = np.random.randint(len(ACTIONS))  # random
        else:
            # Get Q values for each action.
            qval = model.predict(state, batch_size=1)
            action = (np.argmax(qval))  # best
        
        # Take action, observe new state and get rewards. 
        new_state, reward, _, _ = env.step([ACTIONS[action]])
        new_state_array = (np.array([new_state[0], new_state[1]])).reshape((1,2))
        episode_reward += reward
        
        # Experience replay storage.
        replay.append((state_array, action, reward, new_state_array))
        
        # If we're done observing, start training.
        if t > observe:

            # If we've stored enough in our buffer, pop the oldest.
            if len(replay) > buffer:
                replay.pop(0)

            # Randomly sample our experience replay memory
            minibatch = random.sample(replay, batchSize)

            # Get training values.
            X_train, y_train = process_minibatch(minibatch, model)

            # Train the model on this batch.
            history = LossHistory()
            model.fit(
                X_train, y_train, batch_size=batchSize,
                nb_epoch=1, verbose=0, callbacks=[history]
            )
            loss_log.append(history.losses)

        # Update the starting state with S'.
        state = new_state
        state_array = new_state_array
        
        # TODO: Check whether this is required. Decrement epsilon over time.
        if epsilon > 0.1 and t > observe:
            epsilon -= (1/train_frames)
        
        if t % EPISODE_SIZE == 0:
            print 'Obtained reward of %s in episode %d'%(episode_reward, episode_number)
            all_rewards.append(episode_reward)
            episode_reward = 0
            episode_number += 1
            state = env.reset()
            state_array = (np.array([state[0], state[1]])).reshape((1,2))
        
        # Save the model every 25,000 frames.
        if t % 25000 == 0:
            model.save_weights('saved-models/' + filename + '-' +
                               str(t) + '.h5',
                               overwrite=True)
            print("Saving model %s - %d" % (filename, t))

    # Log results after we're done all frames.
    log_results(filename, all_rewards, loss_log)


# In[87]:

def process_minibatch(minibatch, model):
    """This does the heavy lifting, aka, the training. It's super jacked."""
    X_train = []
    y_train = []
    # Loop through our batch and create arrays for X and y
    # so that we can fit our model at every step.
    for memory in minibatch:
        # Get stored values.
        old_state_m, action_m, reward_m, new_state_m = memory
        # Get prediction on old state.
        old_qval = model.predict(old_state_m, batch_size=1)
        # Get prediction on new state.
        newQ = model.predict(new_state_m, batch_size=1)
        # Get our best move. I think?
        maxQ = np.max(newQ)
        y = np.zeros((1, len(ACTIONS)))
        y[:] = old_qval[:]
        # Perform the update. TODO: Check if this update is right.
        update = (reward_m + (GAMMA * maxQ))
        # Update the value for the action we took.
        y[0][action_m] = update
        X_train.append(old_state_m.reshape(NUM_INPUT,))
        y_train.append(y.reshape(len(ACTIONS),))

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train


# In[88]:

def params_to_filename(params):
    return str(params['nn'][0]) + '-' + str(params['nn'][1]) + '-' +             str(params['batchSize']) + '-' + str(params['buffer'])


# In[89]:

def launch_learn(params):
    filename = params_to_filename(params)
    print("Trying %s" % filename)
    # Make sure we haven't run this one.
    if not os.path.isfile('results/sonar-frames/loss_data-' + filename + '.csv'):
        # Create file so we don't double test when we run multiple
        # instances of the script at the same time.
        open('results/sonar-frames/loss_data-' + filename + '.csv', 'a').close()
        print("Starting test.")
        # Train.
        model = neural_net(NUM_INPUT, params['nn'])
        train_net(model, params)
    else:
        print("Already tested.")


# In[90]:

def log_results(filename, episode_rewards, loss_log):
#     # Save the results to a file so we can graph it later.
#     with open('results/neural-net/learn_data-' + filename + '.csv', 'w') as data_dump:
#         wr = csv.writer(data_dump)
#         wr.writerows(data_collect)

    with open('results/neural-net/loss_data-' + filename + '.csv', 'w') as lf:
        wr = csv.writer(lf)
        for loss_item in loss_log:
            wr.writerow(loss_item)

            
    plt.figure()
    plt.plot(episode_rewards)
    plt.savefig('results/neural-net/episodeRewards-' + filename + '.png')
    plt.close()


# In[91]:

if __name__ == "__main__":
    if TUNING:
        param_list = []
        nn_params = [[164, 150], [256, 256],
                     [512, 512], [1000, 1000]]
        batchSizes = [40, 100, 400]
        buffers = [10000, 50000]

        for nn_param in nn_params:
            for batchSize in batchSizes:
                for buffer in buffers:
                    params = {
                        "batchSize": batchSize,
                        "buffer": buffer,
                        "nn": nn_param
                    }
                    param_list.append(params)

        for param_set in param_list:
            launch_learn(param_set)

    else:
        nn_param = [164, 150]
        params = {
            "batchSize": 100,
            "buffer": 50000,
            "nn": nn_param
        }
        model = neural_net(NUM_INPUT, len(ACTIONS), nn_param)
        train_net(model, params)


# 
