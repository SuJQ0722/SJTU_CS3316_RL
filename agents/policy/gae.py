import numpy as np

def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation (GAE-Lambda)
    Parameters:
    - rewards: (T,) ndarray. Rewards from the trajectory.
    - values:  (T,) ndarray. Estimated values V(s_t) from the critic.
    - dones:   (T,) ndarray of bools. Whether each step ends an episode.
    - last_value: float. Estimated V(s_T), for bootstrapping the final step.
    - gamma: float. Discount factor.
    - lam: float. GAE lambda.

    Returns:
    - advantages: (T,) ndarray. Advantage estimates A_t.
    - returns: (T,) ndarray. Return estimates R_t = A_t + V(s_t)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_non_terminal = 1.0 - dones[t]
        next_value = values[t + 1] if t + 1 < T else last_value
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * next_non_terminal * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns