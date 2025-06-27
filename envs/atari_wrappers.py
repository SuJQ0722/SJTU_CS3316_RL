# envs/atari_wrappers.py
# Reference: OpenAI Baselines, Stable-Baselines3

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import cv2
import ale_py

gym.register_envs(ale_py) 
cv2.setNumThreads(0)


class NoopResetEnv(gym.Wrapper):
    """
    When reset the environment, it performs a random number of no-op actions
    (between 1 and noop_max) before returning the first observation.
    This is useful for Atari games where the initial state can vary,
    allowing the agent to start in a more diverse set of initial conditions.
    The no-op action is assumed to be '0' (the first action in the action space).
    """
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0  # 假设动作 '0' 是 No-op
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, _, _, info = self.env.step(self.noop_action)
        return obs, info

class FireResetEnv(gym.Wrapper):
    """
    对于需要按 'FIRE' 键开始游戏或新一轮生命的 Atari 游戏 (如 Breakout)，
    这个 Wrapper 会在重置或失去生命后自动执行 'FIRE' 动作。
    """
    def __init__(self, env):
        super().__init__(env)
        action_meanings = env.unwrapped.get_action_meanings()
        assert action_meanings[1] == 'FIRE'

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, _, _, info = self.env.step(1) # 'FIRE' action
        return obs, info


class EpisodicLifeEnv(gym.Wrapper):
    """
    在训练时，将失去一条生命视为回合的结束 (done=True)。
    这能提供更强的监督信号，因为智能体在犯错（失去生命）后能立即收到负反馈，
    而不是等到游戏完全结束后。
    """
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # 失去生命，但在游戏中还未结束，我们强制认为回合结束
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # 如果上一个回合是真正的游戏结束，才重置环境
        # 否则只是失去了一条命，我们只重置生命计数器
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # 不需要重置，因为环境内部已经处理了新生命的开始
            # (通常由 FireResetEnv 触发)
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info

class MaxAndSkipEnv(gym.Wrapper):
    """
    合并最后两帧图像（取最大值）并跳过指定数量的帧。
    1. 取最大值：消除某些 Atari 游戏中的屏幕闪烁问题。
    2. 跳帧：大幅加快训练速度，并让智能体每次决策都基于一个更长的时间窗口。
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if terminated or truncated:
                break
        
        # 取 buffer 中最后两帧的最大值作为最终观测
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._obs_buffer.clear()
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs, info

class WarpFrame(gym.ObservationWrapper):
    """
    将图像观测进行预处理：
    1. 转换为灰度图。
    2. 缩放到 84x84 的尺寸。
    """
    def __init__(self, env):
        super().__init__(env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
        )

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None] # 添加一个通道维度


def wrap_deepmind(env, episode_life=True, clip_rewards=False, frame_stack=4, scale=False):
    """
    按照 DeepMind 论文的标准流程封装 Atari 环境。
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
        
    env = WarpFrame(env)
    # Note: 其他实现中可能会有 clip_rewards 和 scale wrappers，这里为了简化暂时省略
    # 但 FrameStack 是必须的
    if frame_stack > 0:
        env = gym.wrappers.FrameStackObservation(env, frame_stack)
        
    return env

def make_atari(env_id, max_episode_steps=None, render_mode=None):
    """
    创建并封装一个 Atari 环境的工厂函数。
    """
    env = gym.make(env_id, max_episode_steps=max_episode_steps, render_mode=render_mode)
    # 确保在所有 wrapper 之前先重置一次，避免 NoopResetEnv 的 bug
    env.reset() 
    
    # 修正封装顺序，这很重要
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4) # 先跳帧再处理图像
    
    # wrap_deepmind 内部已经包含了 EpisodicLife, FireReset, WarpFrame, FrameStack, Permute
    env = wrap_deepmind(env, episode_life=True, frame_stack=4)
    
    return env