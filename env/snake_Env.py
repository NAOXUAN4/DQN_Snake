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
        pygame.display.set_caption("🐍")

        self.reset()


    def reset(self):
         
        # 初始化 snake位置数组
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]  # 蛇的初始位置 (环境中心)
        self.direction = Direction.RIGHT  

        # 初始化 food
        self.food = self.place_food()

        # 初始化分数
        self.score = 0

    def step(self, action):
         
        # 获取 action 方向
        self.direction = Direction(action)

        # 更新蛇头方向
        head_x, head_y = self.snake[0]
        if self.direction == Direction.RIGHT:
            new_head = (head_x + 1, head_y)
        elif self.direction == Direction.LEFT:
            new_head = (head_x - 1, head_y)
        elif self.direction == Direction.UP:
            new_head = (head_x, head_y - 1)
        elif self.direction == Direction.DOWN:
            new_head = (head_x, head_y + 1)

        #是否死亡
        isDead = (
            head_x < 0 or head_x >= self.grid_size or
            head_y < 0 or head_y >= self.grid_size or
            (head_x, head_y) in self.snake[1:]
        )
        
        if isDead:
            return self._get_state(), Config.reward["dead"], isDead, {}    # return: state、reward、isDone、other
        
        # 移动蛇
        self.snake.insert(0, new_head)

        # 是否吃到食物
        reward_tmp = 0
        if new_head == self.food:
            self.score += 1
            reward_tmp = Config.reward["eat"]
            self.place_food()  # 重新生成食物
        else:
            self.snake.pop()  # 若没吃到食物，移除尾部
            reward_tmp = Config.reward["step"]

        return self._get_state(), reward_tmp, isDead, {}


    def render(self):
        pass

    def _place_food(self):
        while True:
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            if (x, y) not in self.snake:
                self.food = (x, y)
                break


    def _get_state(self)  -> np.ndarray:
        head_x, head_y = self.snake[0]
        
        danger_straight = self._is_danger(   # x,y + 轴上偏移量
            head_x + (self.direction == Direction.RIGHT) - (self.direction == Direction.LEFT),
            head_y + (self.direction == Direction.DOWN) - (self.direction == Direction.UP)
        )
        danger_right = self._is_danger(     # x,y + 90°偏移量
            head_x + (self.direction == Direction.DOWN) - (self.direction == Direction.UP),
            head_y + (self.direction == Direction.LEFT) - (self.direction == Direction.RIGHT)
        )
        danger_left = self._is_danger(     # x,y + -90°偏移量
            head_x + (self.direction == Direction.UP) - (self.direction == Direction.DOWN),
            head_y + (self.direction == Direction.RIGHT) - (self.direction == Direction.LEFT)
        )
        
        state = [
            head_x/self.grid_size,                    # 蛇头x坐标
            head_y/self.grid_size,                    # 蛇头y坐标
            self.food[0]/self.grid_size,              # 食物x坐标
            self.food[1]/self.grid_size,              # 食物y坐标
            danger_straight,                          # 前方危险
            danger_right,                             # 右方危险
            danger_left,                              # 左方危险
            self.direction == Direction.RIGHT,        # 当前方向
            self.direction == Direction.DOWN,
            self.direction == Direction.LEFT,
            self.direction == Direction.UP
        ]
        
        return np.array(state, dtype=np.float32)
    
    def _is_danger(self, x, y):
        return (x < 0 or x >= self.grid_size or
                y < 0 or y >= self.grid_size or
                (x, y) in self.snake)



    def close(self):
        pygame.quit()




    

        

