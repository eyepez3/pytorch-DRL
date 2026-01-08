import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0
STD_MIN = 0.05

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=3e-4, gamma=0.99):
        super().__init__()

        self.action_dim = action_dim
        self.epsilon = 1e-6
        self.stdmin = 1e-4

        self.base = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Critic
        self.critic = nn.Linear(hidden_dim, 1)

        # Actor
        self.actor_mu = nn.Linear(hidden_dim, action_dim)
        self.actor_std = nn.Linear(hidden_dim, action_dim)

        nn.init.uniform_(self.actor_mu.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.actor_mu.bias)

        nn.init.uniform_(self.actor_std.weight, -1e-3, 1e-3)
        nn.init.constant_(self.actor_std.bias, -0.5)

    def forward(self, state):
        x = self.base(state)

        mu = self.actor_mu(x)
        std = F.softplus(self.actor_std(x)) + self.stdmin

        value = self.critic(x).squeeze(-1)

        return mu, std, value

    def getActorCritic(self, state):
        mu, std, value = self.forward(state)

        if torch.isnan(mu).any() or torch.isnan(std).any():
            raise ValueError(f"NaN detected: mu={mu}, std={std}")

        dist = Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)

        log_prob = dist.log_prob(z)
        log_prob -= torch.log(1 - action.pow(2) + self.epsilon)
        log_prob = log_prob.sum(-1)

        return action, log_prob, value, mu, std