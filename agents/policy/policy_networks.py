import torch
import torch.nn as nn
from torch.distributions import Normal

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64), log_std_init=-0.5):
        super(GaussianPolicy, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # Define the hidden layers
        layers = []
        in_size = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layer for mean and log_std
        self.mean = nn.Linear(in_size, act_dim)
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

    def forward(self, obs):
        # Forward pass through the hidden layers
        x = self.hidden_layers(obs)
        # Compute mean and standard deviation
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std
    
    def get_distribution(self, obs):
        mean, std = self(obs)
        return Normal(mean, std)
    
    def sample(self, obs):
        """Sample an action from the policy given an observation. This instead of argmax ensure the exploration."""
        dist = self.get_distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
    
    def evaluate(self, obs, actions):
        dist = self.get_distribution(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy
    
class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=(64, 64)):
        super(ValueNetwork, self).__init__()
        self.obs_dim = obs_dim
        
        # Define the hidden layers
        layers = []
        in_size = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layer for value
        self.value = nn.Linear(in_size, 1)

    def forward(self, obs):
        x = self.hidden_layers(obs)
        return self.value(x).squeeze(-1)
    
if __name__ == "__main__":
    # Example usage
    obs_dim = 4
    act_dim = 2
    policy = GaussianPolicy(obs_dim, act_dim)
    value_net = ValueNetwork(obs_dim)

    print(policy.hidden_layers.__len__)
    print(value_net.hidden_layers.__len__)
    obs = torch.randn(1, obs_dim)
    action, log_prob = policy.sample(obs)
    value = value_net(obs)

    print("Action:", action)
    print("Log Probability:", log_prob)
    print("Value:", value)
