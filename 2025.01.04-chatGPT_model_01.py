import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

hidden_dim = 256
learning_rate = 1e-4
grad_clip = 0.5
entropy_coef = 0.01  # optional


class TanhGaussianActorCritic(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        min_std=1e-4
    ):
        super().__init__()

        self.min_std = min_std
        self.action_dim = action_dim

        # Shared trunk
        self.base = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor heads
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.std_head = nn.Linear(hidden_dim, action_dim)

        # Critic head
        self.value_head = nn.Linear(hidden_dim, 1)

    # -------------------------------------------------
    # Forward pass (no sampling)
    # -------------------------------------------------
    def forward(self, state):
        x = self.base(state)

        mu = self.mu_head(x)
        raw_std = self.std_head(x)

        # Stable std via Softplus
        std = F.softplus(raw_std) + self.min_std

        value = self.value_head(x)

        return mu, std, value

    # -------------------------------------------------
    # Sample action with tanh squashing
    # -------------------------------------------------
    def sample_action(self, state, eps=1e-6):
        mu, std, value = self.forward(state)

        # Gaussian in R^n
        dist = Normal(mu, std)

        # Reparameterized sample
        z = dist.rsample()

        # Squash to (-1, 1)
        action = torch.tanh(z)

        # Log-probability correction
        log_prob = dist.log_prob(z)
        log_prob -= torch.log(1 - action.pow(2) + eps)
        log_prob = log_prob.sum(-1)

        return action, log_prob, value

    # -------------------------------------------------
    # Deterministic action (for evaluation)
    # -------------------------------------------------
    def act_deterministic(self, state):
        mu, _, _ = self.forward(state)
        return torch.tanh(mu)
    
    # The following is how to update
    action, log_prob, value = model.sample_action(state)

    # advantage = td_target - value.detach()
    policy_loss = -(log_prob * advantage.detach()).mean()
    value_loss = F.mse_loss(value.squeeze(-1), td_target)

    loss = policy_loss + value_loss

# gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)