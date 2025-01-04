import gym
import numpy as np
from env.snake_Env import snakeEnv 
from agent.dqn_agent import DQNAgent

import matplotlib.pyplot as plt
from collections import deque
import time

from snake_config import Config

def plot_scores(scores, mean_scores):
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.pause(0.1)

def train():

    # 初始化
    render_every = Config.render_every

    env = snakeEnv()
    state_size = env.observation_space.shape[0] #11
    action_size = env.action_space.n  #4
    agent = DQNAgent(state_size, action_size)

    # 数据记录
    scores = []
    mean_scores = []
    total_score = 0
    record = 0  # 最高记录# 数据记录

    # 训练
    for e in range(Config.episodes):
        
        state = env.reset()
        isDead = False
        score = 0

        while not isDead:
            # render
            if e % render_every == 0:
                env.render()
                time.sleep(0.1)

            # prediect action 
            action = agent.act(state)  # max(Q_vlaues)

            # take action
            next_state, reward, isDead, _ = env.step(action)

            # store experience
            agent.update(state, action, reward, next_state, isDead)
            score += reward

            # train agent
            loss = agent.train()

            # update state
            state = next_state

            if isDead:
                break

        # 记录得分
        scores.append(score)
        total_score += score
        mean_score = total_score / (e + 1)
        mean_scores.append(mean_score)

        if score > record:
            record = score

        # 打印训练信息
        print(f'Game {e} Score {score} Record {record}')

        # # 更新得分图
        # if episode_now % Config.plot_scores_every == 0:
        #     plot_scores(scores, mean_scores)

    env.close()
    plt.close()






if __name__ == "__main__":



    train()