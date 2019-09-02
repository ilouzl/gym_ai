import keras
from keras import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam, SGD
import gym_2048
import gym
import numpy as np
from collections import deque
import random

env = gym.make('2048-v0')
env.reset()
 
NUM_OF_ACTIONS = env.action_space.n
STATE_SIZE = env.observation_space.shape[0] ** 2
EXPERIENCE_MEMORY_SIZE = 10_000
EXPERIENCE_MEMORY_BATCH_SIZE = 32
TRAIN_PERIOD = 100
UPDATE_TARGET_MODEL_PERIOD = 2
DISCOUNT = 0.99
lr = 0.01

NUM_OF_EPISODES = 20_000

experience_memory = deque(maxlen=EXPERIENCE_MEMORY_SIZE)


def get_model(input_dim, output_dim=1):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, kernel_initializer='random_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, kernel_initializer='random_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, kernel_initializer='random_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, kernel_initializer='random_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, kernel_initializer='random_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim, kernel_initializer='random_uniform'))
    if output_dim == 1:
        model.add(Activation('sigmoid'))
        model.compile(optimizer='rmsprop',
              loss='cate',
              metrics=['accuracy'])
    else:
        model.compile(loss=keras.losses.mse, optimizer='sgd', metrics=["accuracy"])
    return model


policy_model = get_model(input_dim=STATE_SIZE, output_dim=NUM_OF_ACTIONS)
target_model = get_model(input_dim=STATE_SIZE, output_dim=NUM_OF_ACTIONS)
target_model.set_weights(policy_model.get_weights())

def is_final_state(state):
    return (state>0).all()

def preprocess_batch(batch):
    x = []
    y = []
    for b in batch:
        state = b[0]
        new_state = b[3]
        reward = b[2]
        target_q = policy_model.predict(state.reshape(-1,STATE_SIZE)).flatten()
        action = np.argmax(target_q)
        if is_final_state(state):
            max_q_opt = 0
        else:
            max_q_opt = target_model.predict(new_state.reshape(-1,STATE_SIZE)).max()
        target_q[action] = reward + DISCOUNT*max_q_opt
        x.append(state)
        y.append(target_q)
    
    return np.vstack(x), np.vstack(y)

epsilon_step = 0
for ep in range(NUM_OF_EPISODES):
    curr_state = env.reset()
    t = 0
    done = False
    total_reward = 0
    if ep % UPDATE_TARGET_MODEL_PERIOD:
        target_model.set_weights(policy_model.get_weights())
    while not done:

        # act
        epsilon = 0.01 + (1 - 0.01) * np.exp(-1. * epsilon_step * 0.001)
        epsilon_step += 1
        if np.random.rand() < epsilon:
            q = target_model.predict(curr_state.reshape(-1,STATE_SIZE))
            action = np.argmax(q[0])
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)
        total_reward += reward
        exp = (curr_state, action, reward, new_state)
        experience_memory.append(exp)
        done = is_final_state(new_state)
        if not done:
            # if t % TRAIN_PERIOD:
                if len(experience_memory) > 5*EXPERIENCE_MEMORY_BATCH_SIZE:
                    batch = random.sample(experience_memory, EXPERIENCE_MEMORY_BATCH_SIZE)
                    x, y = preprocess_batch(batch)
                    policy_model.fit(x=x.reshape(-1,STATE_SIZE), y=y.reshape(-1,NUM_OF_ACTIONS), verbose=0, batch_size=EXPERIENCE_MEMORY_BATCH_SIZE)
        else:
            print("max tile = %d, total reward = %d, epsilon = %f, at %d"%(curr_state.max(), total_reward, epsilon, t))

        if False:
            print('Next Action: "{}"\n\nReward: {}'.format(
                gym_2048.Base2048Env.ACTION_STRING[action], reward))
            env.render()
        curr_state = new_state
        t += 1





