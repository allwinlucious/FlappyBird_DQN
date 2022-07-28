import numpy as np
import time
import flappy_bird_gym
import torch
from pathlib import Path

if not Path('model/b2d.pt').is_file():
    print ("model doesnt exist , run dqn_train.py first")
    exit()
q_network = torch.load("model/b2d.pt")
policy = q_network
policy.eval()
env= flappy_bird_gym.make("FlappyBird-v0")
for i in range(10):
    env.seed(i)
    state, done = env.reset(), False
    while not done:
        probs = policy(torch.tensor(state).float().reshape((1, -1)))[0]
        action = np.argmax(probs.detach().numpy())  # Greedy action
        state, reward, done, _ = env.step(action)
        env.render()
        time.sleep(1 / 30)  # FPS

env.close()


# In[ ]: