import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import torch.nn as nn
import time

from .model import ActorCritic, compute_gae
from utils.logger import Logger

class PPOAgent:
    """
    一个完全封装的 PPO Agent。
    它管理自己的环境、网络、训练循环和日志记录。
    """
    def __init__(self, config, env_name):
        self.config = config
        
        # --- 初始化环境和核心参数 ---
        self.env_config = config['env']
        self.train_config = config['training']
        self.agent_config = config['agent']
        
        self.run_name = f"ppo-{self.env_config['id']}_{env_name}_{int(time.time())}"
        self.save_dir = os.path.join("models", self.run_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.device = torch.device(self.train_config['device'] if torch.cuda.is_available() else "cpu")
        np.random.seed(self.train_config['seed'])
        torch.manual_seed(self.train_config['seed'])
        
        # Agent 自己创建环境
        self.env = gym.make(env_name)
        
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]

        # --- 初始化网络和优化器 ---
        self.network = ActorCritic(obs_dim, act_dim, tuple(self.agent_config['hidden_sizes'])).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.agent_config['lr'], eps=1e-5)

        # --- 初始化日志 ---
        self.logger = Logger(
            log_dir=self.save_dir,
            project_name='d3qn',
            run_name=self.run_name, 
            config=self.config,
            use_tensorboard=True,
            use_wandb=False
        )
        
        self.global_step = 0

    def save_model(self, step):
        """保存模型权重"""
        model_path = os.path.join(self.save_dir, f"model_step_{step}.pth")
        torch.save(self.network.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path):
        """加载模型权重"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.network.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"Model loaded from {model_path}")

    def _collect_rollout(self, pbar):
        """
        私有方法，用于收集一个完整的 rollout (n_steps)。
        这个方法现在是 Agent 的内部实现细节。
        """
        rollout_data = {
            'obs': [], 'actions': [], 'logps': [], 'rewards': [], 'dones': [], 'values': []
        }
        
        # 如果这是第一次收集，重置环境
        if self.global_step == 0:
            self.obs, _ = self.env.reset(seed=self.train_config['seed'])
            self.ep_return = 0
            self.ep_len = 0
            
        for _ in range(self.agent_config['n_steps']):
            self.global_step += 1
            
            with torch.no_grad():
                obs_tensor = torch.tensor(self.obs, dtype=torch.float32).to(self.device).unsqueeze(0)
                action, logp, _, value = self.network.get_action_and_value(obs_tensor)

            # 存储数据
            rollout_data['obs'].append(self.obs)
            rollout_data['actions'].append(action.cpu().numpy().flatten())
            rollout_data['logps'].append(logp.cpu().item())
            rollout_data['values'].append(value.cpu().item())
            
            # 与环境交互
            next_obs, reward, terminated, truncated, info = self.env.step(action.cpu().numpy().flatten())
            done = terminated or truncated
            
            rollout_data['rewards'].append(reward)
            rollout_data['dones'].append(done)
            
            self.obs = next_obs
            self.ep_return += reward
            self.ep_len += 1

            if done:
                tqdm.write(f"Global Step: {self.global_step}, Episodic Return: {self.ep_return}")
                pbar.set_postfix({"Episodic Return": f"{self.ep_return:.2f}"})
                self.logger.log({
                    "charts/episodic_return": self.ep_return,
                    "charts/episodic_length": self.ep_len,
                }, step=self.global_step)
                self.obs, _ = self.env.reset()
                self.ep_return = 0
                self.ep_len = 0
        
        # 将 list 转换为 numpy array
        for key, val in rollout_data.items():
            rollout_data[key] = np.array(val)
        
        pbar.update(self.agent_config['n_steps'])
        return rollout_data

    def _update(self, trajectories):
        """私有方法，用于执行 PPO 的更新步骤。"""
        # 计算 GAE 和 Returns
        with torch.no_grad():
            last_obs_tensor = torch.tensor(self.obs, dtype=torch.float32).to(self.device).unsqueeze(0)
            last_value = self.network.get_value(last_obs_tensor).cpu().item()

        advantages, returns = compute_gae(
            trajectories['rewards'], trajectories['values'], trajectories['dones'],
            last_value, self.agent_config['gamma'], self.agent_config['gae_lambda']
        )
        
        # 转换为 tensor
        obs_t = torch.tensor(trajectories['obs'], dtype=torch.float32).to(self.device)
        act_t = torch.tensor(trajectories['actions'], dtype=torch.float32).to(self.device)
        logp_t = torch.tensor(trajectories['logps'], dtype=torch.float32).to(self.device)
        adv_t = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        ret_t = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        # 标准化优势
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        
        # 训练循环
        indices = np.arange(self.agent_config['n_steps'])
        for _ in range(self.agent_config['n_epochs']):
            np.random.shuffle(indices)
            for start in range(0, self.agent_config['n_steps'], self.agent_config['batch_size']):
                end = start + self.agent_config['batch_size']
                mb_idx = indices[start:end]

                _, mb_logp, mb_entropy, mb_val = self.network.get_action_and_value(
                    obs_t[mb_idx], act_t[mb_idx]
                )
                
                logratio = mb_logp - logp_t[mb_idx]
                ratio = torch.exp(logratio)
                
                # PPO Clipped Objective
                pg_loss1 = -adv_t[mb_idx] * ratio
                pg_loss2 = -adv_t[mb_idx] * torch.clamp(ratio, 1 - self.agent_config['clip_epsilon'], 1 + self.agent_config['clip_epsilon'])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Function Loss
                v_loss = 0.5 * ((mb_val - ret_t[mb_idx]) ** 2).mean()

                # Entropy Loss
                entropy_loss = mb_entropy.mean()
                
                loss = pg_loss - self.agent_config['ent_coef'] * entropy_loss + self.agent_config['vf_coef'] * v_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.agent_config['max_grad_norm'])
                self.optimizer.step()
        
        # 记录训练指标
        self.logger.log({
            "losses/policy_loss": pg_loss.item(),
            "losses/value_loss": v_loss.item(),
            "losses/entropy": entropy_loss.item(),
        }, step=self.global_step)

    def learn(self):
        """
        公开的 learn 方法，包含了完整的训练循环。
        这是 `run.py` 调用的唯一接口。
        """
        num_updates = self.train_config['total_timesteps'] // self.agent_config['n_steps']
        save_interval = self.train_config.get('save_interval', 50000)
        last_save_step = 0

        pbar = tqdm(total=self.train_config['total_timesteps'], desc="PPO Training")
        while self.global_step < self.train_config['total_timesteps']:
            
            # --- 收集数据 ---
            trajectories = self._collect_rollout(pbar) # 传递 pbar
            
            # --- 更新网络 ---
            self._update(trajectories)

            # --- 定期保存模型 ---
            if self.global_step - last_save_step >= save_interval:
                self.save_model(self.global_step)
                last_save_step = self.global_step
        
        # 训练结束后，再保存一次最终模型
        self.save_model("final")
        pbar.close()
    
    def close(self):
        """关闭环境和日志记录器。"""
        self.env.close()
        self.logger.finish()