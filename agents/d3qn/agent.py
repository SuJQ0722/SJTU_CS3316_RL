import torch
import torch.optim as optim
import numpy as np
import os
import time 
from tqdm import tqdm

from envs.atari_wrappers import make_atari
from utils import ReplayBuffer, LinearScheduler
from .model import D3QN
from utils.logger import Logger

class D3QNAgent:
    def __init__(self, config, env_name):
        self.config = config
        self.env_config = config['env']
        self.agent_config = config['agent']
        self.train_config = config['training']

        self.run_name = f"d3qn-{self.env_config['id']}_{env_name}_{int(time.time())}"
        self.save_dir = os.path.join('models', self.run_name)
        os.makedirs(self.save_dir, exist_ok=True)

        self.device = torch.device(self.train_config['device'] if torch.cuda.is_available() else 'cpu')
        np.random.seed(self.train_config['seed'])
        torch.manual_seed(self.train_config['seed'])

        self.env = make_atari(env_name, self.env_config['frame_stack'])

        self.q_network = D3QN(self.env.observation_space.shape, self.env.action_space.n).to(self.device)
        self.target_q_network = D3QN(self.env.observation_space.shape, self.env.action_space.n).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.agent_config['lr'], eps=1e-6)

        self.replay_buffer = ReplayBuffer(
            self.agent_config['buffer_size'], 
            self.env.observation_space, 
            self.env.action_space, 
            self.device
        )
        self.epsilon_scheduler = LinearScheduler(
            start_value=self.agent_config['eps_start'],
            end_value=self.agent_config['eps_end'],
            duration=int(self.agent_config['eps_fraction'] * self.train_config['total_timesteps'])
        )
        self.logger = Logger(
            log_dir=self.save_dir,
            project_name='d3qn',
            run_name=self.run_name, 
            config=self.config,
            use_tensorboard=True,
            use_wandb=False
        )


    def save_model(self, step):
        """save the model weights to a file"""
        model_path = os.path.join(self.save_dir, f"model_step_{step}.pth")
        torch.save(self.network.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path):
        """load the model weights from a file"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.network.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"Model loaded from {model_path}")

    def _update(self, step):
        data = self.replay_buffer.sample(self.agent_config['batch_size'])

        with torch.no_grad():
            # use online q_network to select actions
            next_q_values_online =  self.q_network(data['next_observations'])
            best_next_actions = next_q_values_online.argmax(dim=1, keepdim=True)

            # use target q_network to compute target q values
            next_q_values_target = self.target_q_network(data['next_observations'])
            next_q_values = next_q_values_target.gather(1, best_next_actions)
            target_q_values = data['rewards'] + (1 - data['dones']) * self.agent_config['gamma'] * next_q_values

        current_q_values = self.q_network(data['observations']).gather(1, data['actions'])
        loss = torch.nn.functional.smooth_l1_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if step % self.agent_config['target_update_interval'] == 0:
            self.logger.log({"losses/q_loss": loss.item()}, step=step)

    def learn(self):
        obs, _ = self.env.reset(seed=self.train_config['seed'])
        ep_return = 0
        ep_len = 0
        
        pbar = tqdm(range(1, self.train_config['total_timesteps'] + 1), desc="D3QN Training")

        for step in pbar:
            # --- Epsilon-Greedy 动作选择 ---
            epsilon = self.epsilon_scheduler.value(step)
            if np.random.rand() < epsilon:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_tensor = torch.tensor(np.array(obs), device=self.device, dtype=torch.float32).unsqueeze(0)
                    q_values = self.q_network(obs_tensor)
                    action = q_values.argmax(dim=1).item()
                    
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            self.replay_buffer.add(obs, next_obs, action, reward, done)

            ep_return += reward
            ep_len += 1

            obs = next_obs

            if done:
                # 真实回合结束时才记录
                if "episode" in info:
                    pbar.set_postfix({"Episodic Return": f"{info['episode']['r']:.2f}"})
                    self.logger.log({
                        "charts/episodic_return": info['episode']['r'],
                        "charts/episodic_length": info['episode']['l'],
                        "charts/epsilon": epsilon,
                    }, step=step)
                obs, _ = self.env.reset()
                ep_return = 0
                ep_len = 0

            # --- 学习更新 ---
            if step > self.agent_config['learning_starts']:
                self._update(step)
                
                # --- 更新 Target Network ---
                if step % self.agent_config['target_update_interval'] == 0:
                    self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        self.save_model("final")

    def close(self):
        self.env.close()
        self.logger.finish()
        