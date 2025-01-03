import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size    # state状态的信息数 例如 state_size = 5 则描述了当前状态的5个属性
        self.action_size = action_size # action种类
        
        # 创建 Q-Network
        self.model = nn.Sequential(       #Q(si,ai)
            nn.Linear(state_size, 64),    # 输入层    state_size -> 64
            nn.ReLU(),                    # 激活函数(Relu)            
            nn.Linear(64, action_size)    # 输出层    64 -> action_size
        )
        
        self.memory = deque(maxlen=2000)  # 经验回放缓冲区
        self.gamma = 0.95    # 折扣因子
        self.epsilon = 1.0   # 探索率
        self.learning_rate = 0.001  # 学习率
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)   #配置Adam优化器  (Momenteum(指数平均) + RMSprop(微分减振))
        print("DQN Agent Initialized\n\n")

    def train(self, batch_size=32) -> float:
        if len(self.memory) < batch_size:   # 等待直到经验池中经验 >= batch_size
            return 0.0
        
        self.model.train()  # 设置为训练模式
            
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)  #解压缩

        # 转换为张量
        states = torch.tensor(states, dtype=torch.float32)             #  shape: (batch_size, state_size)
        actions = torch.tensor(actions, dtype=torch.int64)             #  shape: (batch_size)      
        rewards = torch.tensor(rewards, dtype=torch.float32)           #  shape: (batch_size)
        next_states = torch.tensor(next_states, dtype=torch.float32)   #  shape: (batch_size, state_size)
        dones = torch.tensor(dones, dtype=torch.float32)               #  shape: (batch_size)

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

        # epsilon衰减
        if self.epsilon > 0.01:  # 最小探索率
            self.epsilon *= 0.995  # 衰减因子
        
        return loss.item()


    def act(self, state):  # action
        if random.random() < self.epsilon:  # 探索
            return random.randrange(self.action_size)
        else:  # 对当前状态选择最优动作
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # shape: (1, state_size)
            q_values = self.model(state)  # shape: (1, action_size)
            return q_values.max(1)[1].item()
        

    def update(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


if __name__ == '__main__':

    """
    state = [
    snake_head_x,      # 蛇头x坐标
    snake_head_y,      # 蛇头y坐标
    food_x,           # 食物x坐标
    food_y,           # 食物y坐标
    is_danger_up,     # 上方是否危险
    is_danger_right,  # 右方是否危险
    is_danger_down,   # 下方是否危险
    is_danger_left,   # 左方是否危险
    direction_up,     # 当前方向是否向上
    direction_right,  # 当前方向是否向右
    direction_down    # 当前方向是否向下
    """
    agent = DQNAgent(state_size=5, action_size=6)

    batch_size = 10
    for i in range(batch_size):
        state = np.random.rand(5)
        action = agent.act(state)
        reward, next_state, done = 1, np.random.rand(5), False
        agent.memory.append((state, action, reward, next_state, done))
    agent.train(batch_size=batch_size)
