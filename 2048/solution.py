import gym
import gym_2048
import keras
from keras.utils import to_categorical
import numpy as np

def get_model(input_dim, output_dim=1):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(128, input_dim=input_dim, kernel_initializer='random_uniform'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(256, kernel_initializer='random_uniform'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(512, kernel_initializer='random_uniform'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(256, kernel_initializer='random_uniform'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(128, kernel_initializer='random_uniform'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(output_dim, kernel_initializer='random_uniform'))
    if output_dim == 1:
        model.add(keras.layers.Activation('sigmoid'))
        model.compile(optimizer='rmsprop',
              loss='cate',
              metrics=['accuracy'])
    else:
        model.add(keras.layers.Activation('softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])

    
    
    return model

def extract_data(data):
    data = np.vstack(data)
    state = np.vstack(data[:,0])
    s = np.vstack(data[:,1])

    o = np.vstack(state[:,0])
    a = np.vstack(state[:,1])
    r = np.vstack(state[:,2])
    return {"obs":o, "action":a, "scores":s, "reward":r}

env = gym.make('2048-v0')
env.reset()
observation_size = 16
use_history = True

def random_agent(state):
    return env.action_space.sample()

def agent(state):
    return np.argmax(model.predict(state.reshape(-1,32)))

def play(e, policy, render = False, 
            min_score=50, num_of_episodes=100, 
            episode_len=200, verbose=0):   
    episode_memory = []
    prev_obs = np.zeros(observation_size)
    for i_episode in range(num_of_episodes):
        observation = e.reset().flatten()
        score = 0
        memory = []
        for t in range(episode_len):
            if render:
                e.render()
            
            if use_history:
                s = np.hstack((prev_obs,observation))
            else:
                s = prev_obs
            action = policy(s)
            state = [s,action]

            observation, reward, done, info = env.step(int(action))
            observation = observation.flatten()

            prev_obs = observation

            score = score + reward
            state.append(score)
            memory.append(state)

            if done:
                if verbose > 0:
                    print("Episode {} finished after {} timesteps".format(i_episode, t+1))
                if score > min_score:
                    tmp = [memory, score]
                    episode_memory.append(tmp)
                break
    return episode_memory


training_data = play(env, random_agent, min_score=1000, num_of_episodes=3000)
e = extract_data(training_data)
print("Total %d good episodes" % (len(training_data)))
print("Score Statistics: max = %d, avg = %d" %(e["scores"].max(), e["scores"].mean()))

model = get_model(observation_size + use_history*observation_size, output_dim=env.action_space.n)
model.fit(e["obs"], to_categorical(e["action"]),epochs=5)
test_data = play(env, agent, verbose=1, num_of_episodes=100, min_score=0)
e = extract_data(test_data)
print("Score Statistics: max = %d, avg = %d" %(e["scores"].max(), e["scores"].mean()))
if e["scores"].mean() > 195:
    print("SOLVED!!")
