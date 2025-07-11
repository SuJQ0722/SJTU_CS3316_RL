# agents/ppo/model.py (修正和优化版)

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """对神经网络层进行正交初始化"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    """
    一个集成的 Actor-Critic 网络，共享特征提取层。
    这是一种更高效和优雅的实现方式。
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64), shared=False):
        super().__init__()
        self.shared = shared
        
        if shared:
            # 共享网络结构
            body_layers = []
            in_size = obs_dim
            for h in hidden_sizes:
                body_layers.append(layer_init(nn.Linear(in_size, h)))
                body_layers.append(nn.Tanh()) # 使用 Tanh 激活函数
                in_size = h
            self.actor_critic_body = nn.Sequential(*body_layers)
            self.actor_mean = layer_init(nn.Linear(in_size, act_dim), std=0.01)
            self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
            self.critic_head = layer_init(nn.Linear(in_size, 1), std=1.0)
        else:
            # Policy net 和 Value net 分开
            # Policy network
            policy_layers = []
            in_size = obs_dim
            for h in hidden_sizes:
                policy_layers.append(layer_init(nn.Linear(in_size, h)))
                policy_layers.append(nn.Tanh())
                in_size = h
            self.policy_net = nn.Sequential(*policy_layers)
            self.actor_mean = layer_init(nn.Linear(in_size, act_dim), std=0.01)
            self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
            
            # Value network
            value_layers = []
            in_size = obs_dim
            for h in hidden_sizes:
                value_layers.append(layer_init(nn.Linear(in_size, h)))
                value_layers.append(nn.Tanh())
                in_size = h
            self.value_net = nn.Sequential(*value_layers)
            self.critic_head = layer_init(nn.Linear(in_size, 1), std=1.0)

        
    def get_value(self, obs):
        """获取状态的价值"""
        if self.shared:
            body_out = self.actor_critic_body(obs)
            return self.critic_head(body_out).squeeze(-1)
        else:
            value_out = self.value_net(obs)
            return self.critic_head(value_out).squeeze(-1) 
    
    def get_action_and_value(self, obs, action=None):
        """
        根据观测获取动作、对数概率、熵和价值。
        这个统一的接口非常方便。
        """
        if self.shared:
            # 共享网络结构
            body_out = self.actor_critic_body(obs)
            
            # Actor
            action_mean = self.actor_mean(body_out)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            dist = Normal(action_mean, action_std)
            
            if action is None:
                action = dist.sample()
                
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            
            # Critic
            value = self.critic_head(body_out).squeeze(-1)
            
            return action, log_prob, entropy, value
        else:
            # 分离网络结构
            # Actor
            policy_out = self.policy_net(obs)
            action_mean = self.actor_mean(policy_out)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            dist = Normal(action_mean, action_std)
            
            if action is None:
                action = dist.sample()
                
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            
            # Critic
            value_out = self.value_net(obs)
            value = self.critic_head(value_out).squeeze(-1)
            
            return action, log_prob, entropy, value

    def get_action(self, obs):
        """仅获取动作，用于推理时"""
        if self.shared:
            body_out = self.actor_critic_body(obs)
            action_mean = self.actor_mean(body_out)
        else:
            policy_out = self.policy_net(obs)
            action_mean = self.actor_mean(policy_out)
        
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        dist = Normal(action_mean, action_std)
        return dist.sample()

    
# 修正后的 GAE 函数
def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    T = len(rewards)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae_lam = 0.0
    
    # 将 values 数组扩展，包含 V(s_T)
    values_with_last = np.append(values, last_value)
    
    for t in reversed(range(T)):
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values_with_last[t+1] * next_non_terminal - values_with_last[t]
        advantages[t] = last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
        
    returns = advantages + values
    return advantages, returns