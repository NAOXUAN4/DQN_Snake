import gym
import numpy as np
from env.snake_Env import snakeEnv 
from agent.dqn_agent import DQNAgent

import matplotlib.pyplot as plt
from collections import deque
import time

from snake_config import Config

def predict(load_checkpoint):

    # 初始化
    env = snakeEnv()
    state_size = env.observation_space.shape[0] #11
    action_size = env.action_space.n  #4
    agent = DQNAgent(state_size, action_size)

    # 数据记录
    record = 0  # 最高记录

    # 加载checkpoint
    Model_episode = 0
    if load_checkpoint:
        Model_episode, best_score, _ = agent.load_model(load_checkpoint)
        print(f"Loaded model from episode {Model_episode} with score {best_score}")


    for e in range(10000):

        isDead = False
        state = env.reset()
        score = 0

        while not isDead:
            
            env.render()
            time.sleep(0.1)


            action = agent.act(state, epsilon=0)
            next_state, reward, isDead, _ = env.step(action)
            state = next_state

            score += reward


            if isDead:
                break

        if score > record:
                record = score
        print(f"ep : {e}  Score: {score} record: {record}")

    env.close()




if __name__ == '__main__':
    predict(load_checkpoint = "checkpoints/best.pth")
