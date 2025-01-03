from gym import spaces
import numpy as np

class Config:

      "----------------------------------game parameters-----------------------------------"
      grid_size = 20  # 游戏网格大小
      window_size = 500    # 窗口大小
      cell_size = window_size / grid_size  # 单元格大小

      #动作空间
      action_space = spaces.Discrete(4)

      # 观察空间
      observation_space = spaces.Box(  # 定义 state的 11 个属性 数值范围 float(0~1)
            low=0, high=1, 
            shape=(11,), 
            dtype=np.float32
            )
      
      # 奖励
      reward = {
            "eat": 10,
            "dead": -10,
            "step": -0.1
      }

      # 颜色
      color = {  
            "snake": (0, 255, 0),
            "food": (255, 0, 0),
            "wall": (0, 0, 0)
      }
 

      "----------------------------------train parameters----------------------------------"
      episodes = 1000 # 训练次数
      render_every = 100  # 每n轮显示一次游戏画面
      plot_scores_every = 10  # 每n轮更新一次得分图


      "----------------------------------model parameters----------------------------------"
      batch_size = 32 # Reward buffer 批量大小
      gamma = 0.95
      epsilon = 1.0   # 探索率
      learning_rate = 0.001  # 学习率

      


    
