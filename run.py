import argparse
import torch
import os
import random
import numpy as np
from agents.policy.ppo import PPOAgent

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='Pendulum-v1',type=str,help='Environment name')
    parser.add_argument('--method', type=str, default='ppo', help='The RL method you want to use for training')
    parser.add_argument('--device', type=str, default='cpu', help='cpu 或 cuda')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--train-steps', type=int, default=int(5e6), help='total steps for training')
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

method2class = {
    'ppo': PPOAgent,
}



if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    method = method2class[args.method.lower()]
    try:
        agent = method(args.env, args.device)
    except Exception as e:
        raise ValueError(f"Failed to create agent: {e}")

    print(f"Start training {args.method} on {args.env} using {args.device}")
    agent.train(total_steps=args.train_steps)
