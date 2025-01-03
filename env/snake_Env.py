import gym
import numpy as np
from gym import spaces
import pygame
from enum import Enum

from snake_config import Config

class Direction(Enum):
        UP = 0
        RIGHT = 1
        DOWN = 2
        LEFT = 3

class snakeEnv(gym.Env):
    def __init__(self, grid_size=20):
        super(snakeEnv, self).__init__()

        # 初始化环境参数
        self.grid_size = Config.grid_size        #从配置文件中读取
        self.window_size = Config.window_size
        self.action_space = Config.action_space
        self.observation_space = Config.observation_space

        #初始化 pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))

        

