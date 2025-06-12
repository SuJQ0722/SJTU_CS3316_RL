from importlib.metadata import distributions

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
from .gae import compute_gae
from .policy_networks import GaussianPolicy, ValueNetwork
from tqdm import tqdm

class PPOAgent:
    def __init__(self,
                 env_name,
                 device='cuda',
                 hidden_sizes=(64,64),
                 lr=3e-4,
                 clip_epsilon=0.2,
                 epochs=10,
                 batch_size=64,
                 gamma=0.99,
                 lam=0.95,
                 max_grad_norm=0.5,
                 ent_coef=0.0,
                 vf_coef=0.5
                 ):
        self.env = gym.make(env_name)
        obs_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            act_dim = self.env.action_space.n
        else:
            act_dim = self.env.action_space.shape[0]

        self.device = device
        self.policy = GaussianPolicy(obs_dim, act_dim, hidden_sizes).to(device)
        self.value = ValueNetwork(obs_dim, hidden_sizes).to(device)
        self.optimizer = optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()), lr=lr)

        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

    def collect_trajectories(self, horizen):
        """
        collect horizen steps in a single environment.
        :param horizen:
        :return:
        """
        obs_list, act_list, logp_list, reward_list, value_list, done_list = [], [], [], [], [], []
        obs, info = self.env.reset()
        done = False
        for _ in range(horizen):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
            dist = self.policy.get_distribution(obs_tensor)
            action = dist.sample().cpu().numpy()
            logp = dist.log_prob(torch.tensor(action, dtype=torch.float32).to(self.device))
            value = self.value(obs_tensor).item()

            next_obs, reward, done, _, _ = self.env.step(action)

            obs_list.append(obs)
            act_list.append(action)
            logp_list.append(logp.detach().cpu().item())
            reward_list.append(reward)
            value_list.append(value)
            done_list.append(done)

            obs = next_obs
            if done:
                obs = self.env.reset()

        last_val = self.value(torch.tensor(obs, dtype=torch.float32)).to(self.device).item()
        return {
            'obs': np.array(obs_list),
            'acts': np.array(act_list),
            'logps': np.array(logp_list),
            'rewards': np.array(reward_list),
            'values': np.array(value_list),
            'dones': np.array(done_list),
            'last_val': last_val
        }

    def update(self, trajectories):
        obs = trajectories['obs']
        acts = trajectories['acts']
        logps_old = trajectories['logps']
        rewards = trajectories['rewards']
        values = trajectories['values']
        dones = trajectories['dones']
        last_val = trajectories['last_val']

        advantages, returns = compute_gae(rewards, values, dones, last_val, self.gamma, self.lam)

        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        acts_tensor = torch.tensor(acts, dtype=torch.float32).to(self.device)
        old_logp_tensor = torch.tensor(logps_old, dtype=torch.float32).to(self.device)
        adv_tensor = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        ret_tensor = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # ppo gradient ascent
        dataset_size = len(obs)
        indices = np.arange(dataset_size)
        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                mb_idx = indices[start:end]

                mb_obs = obs_tensor[mb_idx]
                mb_acts = acts_tensor[mb_idx]
                mb_old_logp = old_logp_tensor[mb_idx]
                mb_adv = adv_tensor[mb_idx]
                mb_ret = ret_tensor[mb_idx]

                # calculate logp and value
                dist = self.policy.get_distribution(mb_obs)
                mb_logp = dist.log_prob(mb_acts)
                mb_entropy = dist.entropy().sum(axis=-1).mean()
                mb_val = self.value(mb_obs).squeeze(-1)

                ratio = torch.exp(mb_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1+ self.clip_epsilon) * mb_adv
                loss_actor = -torch.min(surr1, surr2).mean()
                loss_critic = (mb_val - mb_ret).pow(2).mean()
                loss = loss_actor + self.vf_coef * loss_critic - self.ent_coef * mb_entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.value.parameters()), self.max_grad_norm)
                self.optimizer.step()

    def train(self, total_steps, horizon=2048):
        step_counter = 0
        with tqdm(total=total_steps) as pbar:
            while step_counter < total_steps:
                traj = self.collect_trajectories(horizon)
                self.update(traj)
                step_counter += horizon
                pbar.update(horizon)
                pbar.set_postfix({"Last V(s)": f"{traj['values'][-1]:.3f}"})

