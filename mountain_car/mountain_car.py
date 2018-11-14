#!/Users/liron/miniconda3/envs/gym/bin/python
import gym
import keras
import numpy as np

def get_model(input_dim, output_dim=2):
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
    model.add(keras.layers.Softmax())

    model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model

def extract_data(data):
    data = np.vstack(data)
    state = np.vstack(data[:,0])
    s = np.vstack(data[:,1])

    o = np.vstack(state[:,0])
    a = np.vstack(state[:,1])
    r = np.vstack(state[:,2])
    return {"obs":o, "action":a, "scores":s, "reward":r}

env = gym.make('MountainCar-v0')
state_history = 2
n_observation = env.observation_space.sample().size
n_actions = env.action_space.n
n_state = n_observation*(1 + state_history)


model = get_model(input_dim=n_state, output_dim=n_actions)

def random_agent(state):
    return env.action_space.sample()

def agent(state):
    return np.argmax(model.predict(state.reshape(-1,n_state)))

def play(e, policy, render = False, 
            max_score=-150, num_of_episodes=1000, 
            episode_len=500, verbose=0):   
    episode_memory = []
    obs_history = np.zeros((1,int(state_history*n_observation))).flatten()
    for i_episode in range(num_of_episodes):
        observation = e.reset()
        score = 0
        memory = []
        for t in range(episode_len):
            if render:
                e.render()
            
            s = np.hstack((obs_history,observation))
            action = policy(s)
            state = [s,action]

            observation, reward, done, info = env.step(int(action))

            obs_history = np.hstack((obs_history[n_observation:],observation))

            my_score = my_score + observation[0].abs()

            score = score + reward
            state.append(score)
            memory.append(state)

            # if done:
            if observation[1] > 0:
                if verbose > 0:
                    print("Episode: {}, steps: {}, score: {}".format(i_episode, t+1, score))
                if score > max_score:
                    tmp = [memory, score]
                    episode_memory.append(tmp)
                break
    return episode_memory



training_data = play(env, random_agent, max_score=-200, num_of_episodes=3000, verbose=1)
e = extract_data(training_data)
print("Total %d good episodes" % (len(training_data)))
print("Score Statistics: max = %d, avg = %d" %(e["scores"].max(), e["scores"].mean()))

model.fit(e["obs"], e["action"],epochs=2)

test_data = play(env, agent, verbose=1, num_of_episodes=100, min_score=0)
e = extract_data(test_data)
print("Score Statistics: max = %d, avg = %d" %(e["scores"].max(), e["scores"].mean()))
if e["scores"].mean() > -110:
    print("SOLVED!!")
