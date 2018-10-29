# Click Run Above to Start Training the Model
# After training, the best policy will automatically replay on the
# browser on the right.

import gym
import numpy as np

env = gym.make('CartPole-v1')

def play(env, policy):
    observation = env.reset()
    
    done = False
    score = 0
    observations = []
    
    for _ in range(5000):
        observations += [observation.tolist()] # Record the observations for normalization and replay
        
        if done: # If the simulation was over last iteration, exit loop
            break
            
        # Pick an action according to the policy matrix
        outcome = np.dot(policy, observation)
        action = 1 if outcome > 0 else 0
        
        # Make the action, record reward
        observation, reward, done, info = env.step(action)
        score += reward

    return score, observations

print('💪💪💪 Training Policy... \n')

max = (0, [], [])

for _ in range(100):
    policy = np.random.rand(1,4) - 0.5
    score, observations = play(env, policy)
  
    if score > max[0]:
        max = (score, observations, policy)

print('Max Score', max[0], 'out of 500 \n')
print(max[2])
scores = []
for _ in range(100):
    env.render()
    score, _  = play(env, max[2])
    scores += [score]
  
print('Average Score (100 trials)', np.mean(scores))
