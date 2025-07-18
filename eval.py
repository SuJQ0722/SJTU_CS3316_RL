# evaluate.py (支持视频录制)

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import numpy as np
import argparse
import yaml
import os

from agents.ppo.model import ActorCritic
from agents.d3qn.model import D3QN
from envs.atari_wrappers import make_atari

def evaluate_agent(config, env_name, model_path, num_episodes=10, render=False, record=False):
    """
    加载并评估一个已训练的 PPO Agent。

    Parameters:
    - config: 包含环境和 Agent 设置的字典。
    - model_path: 已保存模型权重的路径 (.pth文件)。
    - num_episodes: 要评估的回合数。
    - render: 是否实时渲染环境 (需要图形界面)。
    - record: 是否将评估过程录制成视频。
    """
    env_config = config['env']
    agent_config = config['agent']
    algo = config['algo']['name']
    render_mode = "human" if render else "rgb_array" if record else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if env_name in ['Hopper-v4', 'Ant-v4', 'Humanoid-v4', 'HalfCheetah-v4']:
        env = gym.make(env_name, render_mode=render_mode)
        network = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0], tuple(agent_config['hidden_sizes'])).to(device)
    elif env_name in ['ALE/Breakout-v5', 'ALE/Boxing-v5', 'ALE/Pong-v5', 'ALE/VideoPinball-v5']:
        env = make_atari(env_name, max_episode_steps=env_config['max_episode_steps'], render_mode=render_mode)
        network = D3QN(env.observation_space.shape, env.action_space.n).to(device)
    else:
        raise ValueError(f"Unsupported environment: {env_name}")

    if record:
        video_folder = os.path.join("videos", os.path.basename(os.path.dirname(model_path)))
        os.makedirs(video_folder, exist_ok=True)
        env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: e % 1 == 0, name_prefix="eval")
        print(f"Recording videos to: {video_folder}")

    # load the model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    network.load_state_dict(torch.load(model_path, map_location=device))
    network.eval()

    print(f"Evaluating model from {model_path} for {num_episodes} episodes...")

    # run the evaluation loop
    total_rewards = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
                if algo == 'PPO':
                    action = network.get_action_and_value(obs_tensor)[0].cpu().numpy().flatten()
                elif algo == 'D3QN':
                    action = network(obs_tensor).argmax(dim=1).item()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}")

    env.close()

    # report results
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    
    print("\n--- Evaluation Finished ---")
    print(f"Average Reward over {num_episodes} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")
    print("--------------------------")
    return mean_reward, std_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained RL agent.")
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to the config file used for training")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the trained model .pth file")
    parser.add_argument("--env_name", type=str, required=True,
                        help="Environment name to use for evaluation (default: Hopper-v4)")
    parser.add_argument("--episodes", type=int, default=10, 
                        help="Number of episodes to run for evaluation")
    parser.add_argument("--render", action="store_true", 
                        help="Render the environment in real-time (requires a display)")
    # 新增 --record 参数
    parser.add_argument("--record", action="store_true",
                        help="Record the evaluation to a video file")

    args = parser.parse_args()

    # 加载配置文件
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 渲染和录制是互斥的，但为了简单起见，我们允许同时设置
    evaluate_agent(config, args.env_name, args.model_path, args.episodes, args.render, args.record)