import gym
import numpy as np

env = gym.make('MountainCar-v0')

render = False
N_QUANTIZATION = 20
epsilon = 0

lr = 0.3
discount = 0.8

def quantize_state(s):
    quantized_state = ((s - env.observation_space.low)/(env.observation_space.high - env.observation_space.low)*N_QUANTIZATION).astype(int)
    return tuple(quantized_state)


q_table_shape = (N_QUANTIZATION, N_QUANTIZATION, env.action_space.n)
q_table = np.random.rand(N_QUANTIZATION, N_QUANTIZATION, env.action_space.n)

q_table_history = []

for i_episode in range(3000):
    observation = env.reset()
    for t in range(1000):
        quantized_observation = quantize_state(observation)

        if np.random.random() > epsilon:
            action = np.argmax(q_table[quantized_observation])
        else:
            action = env.action_space.sample()
        new_observation, reward, done, info = env.step(action)
        quantized_new_observation = quantize_state(new_observation)

        if new_observation[0] >= env.goal_position:
                reward = 2*(1 - t/200)
        q_table[quantized_observation][action] += lr * (reward + discount * np.max(q_table[quantized_new_observation]) - q_table[quantized_observation][action])
      
        if done:
            if new_observation[0] >= env.goal_position:
                # q_table[quantized_observation][action] = 0
                reward = t/200
                print("Episode {} finished after {} timesteps".format(i_episode,t+1))
                # render = True
            else:
                render = False
            q_table_history.append(q_table)
            break
    
        if render:
            env.render()
        observation = new_observation