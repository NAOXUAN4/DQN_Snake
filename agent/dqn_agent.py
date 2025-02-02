import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque
import os

from sympy.physics.paulialgebra import epsilon

from snake_config import Config


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size    # state状态的信息数 例如 state_size = 5 则描述了当前状态的5个属性
        self.action_size = action_size # action种类
        
        # 创建 Q-Network
        self.model = nn.Sequential(       #Q(si,ai)
            nn.Linear(state_size, 64),    # 输入层    state_size -> 64
            nn.ReLU(),                    # 激活函数(Relu)            
            nn.Linear(64, action_size)    # 输出层    64 -> action_size
        ).to(Config.device)
        
        self.memory = deque(maxlen=2000)  # 经验回放缓冲区
        self.gamma = Config.gamma       # 折扣因子
        self.learning_rate = Config.learning_rate  # 学习率
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)   #配置Adam优化器  (Momenteum(指数平均) + RMSprop(微分减振))
        print("\nDQN Agent Initialized\n")

    def train(self, batch_size = Config.batch_size) -> float:
        if len(self.memory) < batch_size*1.5:   # 等待直到经验池中经验 >= batch_size
            return 0.0
        
        self.model.train()  # 设置为训练模式
            
        # 获取随机批次
        batch = random.sample(self.memory, batch_size)
        
        batch = list(zip(*batch))  # 解压批次数据

        states = torch.FloatTensor(np.array(batch[0])).to(Config.device)      # [batch_size, state_size] 
        actions = torch.LongTensor(batch[1]).to(Config.device)                # [batch_size]
        rewards = torch.FloatTensor(batch[2]).to(Config.device)                # [batch_size]
        next_states = torch.FloatTensor(np.array(batch[3])).to(Config.device)  # [batch_size, state_size]
        dones = torch.FloatTensor(batch[4]).to(Config.device)   

        q_values = self.model(states)                                  # state_size - > action_size (所有动作对应的reward表)   shape: (batch_size, action_size)
        next_q_values = self.model(next_states)                        # state_size - > action_size (所有动作在next_state下对应的reward表)   shape: (batch_size, action_size)
        
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # 从q_values中取出actions对应的reward值( q_value[i] = q_values[i][actions[i]] )  shape: (batch_size)
        next_q_value = next_q_values.max(1)[0]                         # 取出next_q_values中最大的reward值  shape: (batch_size)
        

        #Q(si,ai) = r + gamma * max(Q(si+1,ai+1))  (Q-learning)
        target_q_value = rewards + self.gamma * next_q_value.detach() * (1 - dones)  # 计算目标Q值   shape: (batch_size)
        
        loss = nn.MSELoss()(q_value, target_q_value)      # loss = Σ((Q(si,ai) - target_q_value)^2) / batch_size (MSE) 

        self.optimizer.zero_grad()  # 梯度清零
        loss.backward()             # 反向传播
        self.optimizer.step()       # 更新参数

        
        return loss.item()


    def act(self, state, epsilon):  # action
        if random.random() < epsilon:  # 探索
            return random.randrange(self.action_size)
        else:  # 对当前状态选择最优动作
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(Config.device)  # shape: (1, state_size)
            q_values = self.model(state)  # shape: (1, action_size)
            return q_values.max(1)[1].item()
        

    def update(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def save_model(self, episode, record, epsilon_):
        
        # 读取模型保存路径
        checkpoint_dir = Config.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)


        checkpoint = {
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': epsilon_,
            'record': record
        }

        path = os.path.join(checkpoint_dir, f'best.pth')
        torch.save(checkpoint, path)
        
    def load_model(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint['episode'], checkpoint['record'],checkpoint['epsilon']
        return 0,0,0
