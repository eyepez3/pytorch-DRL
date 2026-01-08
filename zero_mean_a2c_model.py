import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0
STD_MIN = 0.05


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
        )

        # Actor
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)

        # Critic
        self.value_head = nn.Linear(hidden, 1)

        # Initialize mean VERY small
        nn.init.uniform_(self.mu_head.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.mu_head.bias)

        nn.init.uniform_(self.log_std_head.weight, -1e-3, 1e-3)
        nn.init.constant_(self.log_std_head.bias, -0.5)

    def forward(self, obs):
        x = self.shared(obs)

        # ---- ACTOR ----
        raw_mu = self.mu_head(x)

        # Soft bounding of mean (prevents railing)
        mu = 2.0 * torch.tanh(raw_mu / 2.0)

        raw_log_std = self.log_std_head(x)
        log_std = torch.clamp(raw_log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std) + STD_MIN

        dist = Normal(mu, std)

        # ---- CRITIC ----
        value = self.value_head(x).squeeze(-1)

        return dist, value