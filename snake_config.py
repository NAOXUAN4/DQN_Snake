from gym import spaces
import pygame

class Config:

    "game parameters"
    game_grid_size = 20  # 游戏网格大小
    window_size = 500    # 窗口大小
    cell_size = window_size / game_grid_size  # 单元格大小

    #动作空间
    action_space = spaces.Discrete(4)

    # 观察空间
    observation_space = spaces.Box(  # 定义 state的 11 个属性 数值范围 float(0~1)
          low=0, high=1, 
          shape=(11,), 
          dtype=pygame.float32
          )
 

    "agent parameters"
    
