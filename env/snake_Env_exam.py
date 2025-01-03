import gym
from gym import spaces
import numpy as np
from enum import Enum
import pygame

class Direction(Enum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

class SnakeEnv(gym.Env):
    def __init__(self, grid_size=20):
        super(SnakeEnv, self).__init__()
        
        # 初始化参数
        self.grid_size = grid_size
        self.window_size = 400
        self.cell_size = self.window_size // self.grid_size
        
        # 动作空间 (0:右, 1:下, 2:左, 3:上)
        self.action_space = spaces.Discrete(4)
        
        # 观察空间 (11个特征)
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(11,), 
            dtype=np.float32
        )
        
        # 初始化pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption('Snake DQN')
        
        self.reset()
    
    def reset(self):
        # 初始化蛇的位置和方向
        self.snake = [(self.grid_size//2, self.grid_size//2)]
        self.direction = Direction.RIGHT
        
        # 放置食物
        self.place_food()
        
        # 初始化分数
        self.score = 0
        
        return self._get_state()
    
    def step(self, action):
        # 更新蛇的方向
        self.direction = Direction(action)
        
        # 获取蛇头位置
        head_x, head_y = self.snake[0]
        
        # 根据方向移动蛇头
        if self.direction == Direction.RIGHT:
            head_x += 1
        elif self.direction == Direction.LEFT:
            head_x -= 1
        elif self.direction == Direction.DOWN:
            head_y += 1
        elif self.direction == Direction.UP:
            head_y -= 1
            
        # 检查是否撞墙或撞到自己
        done = (
            head_x < 0 or head_x >= self.grid_size or
            head_y < 0 or head_y >= self.grid_size or
            (head_x, head_y) in self.snake[1:]
        )
        
        if done:
            return self._get_state(), -10, done, {}
            
        # 移动蛇
        self.snake.insert(0, (head_x, head_y))
        
        # 检查是否吃到食物
        reward = 0
        if (head_x, head_y) == self.food:
            self.score += 1
            reward = 10
            self.place_food()
        else:
            self.snake.pop()
            reward = -0.1  # 小惩罚促进蛇主动寻找食物
            
        return self._get_state(), reward, done, {}
    
    def render(self, mode='human'):
        self.screen.fill((0, 0, 0))
        
        # 绘制蛇
        for segment in self.snake:
            pygame.draw.rect(self.screen, (0, 255, 0),
                           (segment[0]*self.cell_size, 
                            segment[1]*self.cell_size,
                            self.cell_size-1, 
                            self.cell_size-1))
            
        # 绘制食物
        pygame.draw.rect(self.screen, (255, 0, 0),
                        (self.food[0]*self.cell_size,
                         self.food[1]*self.cell_size,
                         self.cell_size-1,
                         self.cell_size-1))
                         
        pygame.display.flip()
    
    def _get_state(self):
        head_x, head_y = self.snake[0]
        
        # 计算危险区域
        danger_straight = self._is_danger(
            head_x + (self.direction == Direction.RIGHT) - (self.direction == Direction.LEFT),
            head_y + (self.direction == Direction.DOWN) - (self.direction == Direction.UP)
        )
        danger_right = self._is_danger(
            head_x + (self.direction == Direction.DOWN) - (self.direction == Direction.UP),
            head_y + (self.direction == Direction.LEFT) - (self.direction == Direction.RIGHT)
        )
        danger_left = self._is_danger(
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
    
    def place_food(self):
        while True:
            food = (
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size)
            )
            if food not in self.snake:
                self.food = food
                break

    def close(self):
        pygame.quit()