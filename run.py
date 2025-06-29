import argparse
import yaml
import os

# 动态导入 Agent
from agents.ppo.agent import PPOAgent
from agents.d3qn.agent import D3QNAgent

ALGO2AGENTS = {
    'PPO': PPOAgent,
    'D3QN': D3QNAgent,
}

def main():
    """
    通用训练启动脚本。
    根据命令行参数加载配置并启动相应的 RL Agent 进行训练。
    """
    parser = argparse.ArgumentParser(description="Run RL algorithms")
    parser.add_argument("--env_name", type=str, required=True, help="The environment to run.")
    parser.add_argument("--report_to", type=str, default="wandb", help="Reporting tool (default: wandb)")
    args = parser.parse_args()

    if args.env_name in ['Hopper-v4', 'Ant-v4', 'Humanoid-v4', 'HalfCheetah-v4']:
        config_path = 'configs/ppo.yaml'
    elif args.env_name in ['ALE/Breakout-v5', 'ALE/Boxing-v5', 'ALE/Pong-v5', 'ALE/VideoPinball-v5']:
        config_path = 'configs/d3qn.yaml'
    else:
        raise ValueError(f"Unknown environment: {args.env_name}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    AGENT = ALGO2AGENTS[config['algo']['name']]
    agent = AGENT(config=config, env_name=args.env_name)

    print(f"==================== Starting training {config['algo']['name']} on {args.env_name.upper()} using {config['training']['device']} ====================")
    agent.learn()
    
    print("==================== Training finished ====================")
    agent.close()

if __name__ == "__main__":
    main()