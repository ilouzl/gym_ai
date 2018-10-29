#!/Users/liron/miniconda3/envs/gym/bin/python
import gym
import keras
import numpy as np

def get_model(input_dim):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(200, input_dim=input_dim, kernel_initializer='random_uniform'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(200, input_dim=input_dim, kernel_initializer='random_uniform'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(10, kernel_initializer='random_uniform'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(1, kernel_initializer='random_uniform'))
    model.add(keras.layers.Activation('sigmoid'))

    model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
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

env = gym.make('CartPole-v0')
model = get_model(8)

def random_agent(state):
    return env.action_space.sample()

def agent(state):
    return np.round(model.predict(state.reshape(-1,8)))

def play(e, policy, render = False, 
            min_score=50, num_of_episodes=400, 
            episode_len=200):   
    episode_memory = []
    prev_obs = np.asarray([0,0,0,0])
    for i_episode in range(num_of_episodes):
        observation = e.reset()
        score = 0
        memory = []
        for t in range(episode_len):
            if render:
                e.render()
            
            s = np.hstack((prev_obs,observation))
            action = policy(s)
            state = [s,action]

            observation, reward, done, info = env.step(int(action))

            prev_obs = observation

            score = score + reward
            state.append(score)
            memory.append(state)

            if done:
                print("Episode {} finished after {} timesteps".format(i_episode, t+1))
                if score > min_score:
                    tmp = [memory, score]
                    episode_memory.append(tmp)
                break
    return episode_memory


min_score = 40
for _ in range(10):
    training_data = play(env, agent, min_score=min_score)
    if training_data != []:
        e = extract_data(training_data)
        print("Total %d good episodes" % (len(training_data)))
        print("Score Statistics: max = %d, avg = %d" %(e["scores"].max(), e["scores"].mean()))
        min_score += 10
        model.fit(e["obs"], e["action"],epochs=1)
    else:
        model = get_model(4)