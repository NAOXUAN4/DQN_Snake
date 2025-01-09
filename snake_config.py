from gym import spaces
import numpy as np
import torch

class Config:
      "----------------------------------device---------------------------------------------"
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            "eat": 20,
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
      save_every = 100  # 每 n 轮保存一次
      checkpoint_dir = "checkpoints"  # 保存 checkpoint 的名称
      checkpoint_path = None # 训练时加载的 checkpoint

      episodes = 20000 # 训练次数
      render_every = 100  # 每n轮显示一次游戏画面
      plot_scores_every = 100  # 每n轮更新一次得分图


      "----------------------------------model parameters----------------------------------"
      batch_size = 32 # Reward buffer 批量大小
      gamma = 0.95
      learning_rate = 0.001  # 学习率

      epsilon_default = 1.0 # 探索率默认值
      epsilon_decay = 0.995 # 探索率衰减
      epsilon_min = 0.01

      


    
