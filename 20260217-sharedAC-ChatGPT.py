import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Shared Actor-Critic Network
# ============================================================

class SharedActorCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Actor head
        self.mu = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(1))

        # Critic head
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = self.shared(state)

        mu = torch.tanh(self.mu(x))  # steering bounded [-1,1]
        std = torch.exp(self.log_std).clamp(1e-3, 2.0)

        value = self.value(x)

        return mu, std, value


# ============================================================
# Rollout Buffer (Shared Across All Agents)
# ============================================================

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)


# ============================================================
# Agent Wrapper
# ============================================================

class A2CTrainer:
    def __init__(self, state_dim, lr=3e-4, gamma=0.99, entropy_coef=0.01):
        self.model = SharedActorCritic(state_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.buffer = RolloutBuffer()

    # --------------------------------------------------------
    # Action Selection (called per agent per step)
    # --------------------------------------------------------
    def select_action(self, state_np):
        state = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(device)

        mu, std, value = self.model(state)
        dist = Normal(mu, std)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        action_clamped = action.clamp(-1.0, 1.0)

        return (
            action_clamped.item(),
            log_prob.squeeze(0),
            value.squeeze(0)
        )

    # --------------------------------------------------------
    # Store transition (called per agent per step)
    # --------------------------------------------------------
    def store(self, state, action, log_prob, reward, done, value):
        self.buffer.add(state, action, log_prob, reward, done, value)

    # --------------------------------------------------------
    # Update after rollout (all agents combined)
    # --------------------------------------------------------
    def update(self, last_value=0.0):

        states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32).to(device)
        actions = torch.tensor(self.buffer.actions, dtype=torch.float32).unsqueeze(1).to(device)
        log_probs_old = torch.stack(self.buffer.log_probs).to(device)
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(self.buffer.dones, dtype=torch.float32).to(device)
        values = torch.stack(self.buffer.values).squeeze(-1).to(device)

        # ----------------------------------------------------
        # Compute TD targets
        # ----------------------------------------------------
        returns = []
        R = torch.tensor(last_value).to(device)

        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)

        returns = torch.stack(returns).detach()

        # ----------------------------------------------------
        # Advantage
        # ----------------------------------------------------
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        # ----------------------------------------------------
        # Forward pass again
        # ----------------------------------------------------
        mu, std, new_values = self.model(states)
        dist = Normal(mu, std)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # ----------------------------------------------------
        # Losses
        # ----------------------------------------------------
        actor_loss = -(log_probs.squeeze() * advantages.detach()).mean()
        critic_loss = 0.5 * (returns - new_values.squeeze()).pow(2).mean()
        loss = actor_loss + critic_loss - self.entropy_coef * entropy

        # ----------------------------------------------------
        # Optimize
        # ----------------------------------------------------
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        self.buffer.clear()

        return {
            "loss": loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item()
        }