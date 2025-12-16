"""
A2C (Advantage Actor-Critic) with 1-step TD for continuous action spaces.
Suitable for environments like Pendulum-v1, MountainCarContinuous-v0, etc.

A2C update every step:
    V_target = r + γ * V(s')
    Advantage = V_target - V(s)
    Actor_loss  = -logπ(a|s) * Advantage - entropy_coef * entropy
    Critic_loss = mse(V(s), V_target)

Author: ChatGPT
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
#  Actor Network: Gaussian Policy
# ------------------------------------------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mean = nn.Linear(hidden, action_dim)

        # log_std is trainable. One value per action dim.
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy


# ------------------------------------------------------------
# Critic Network: State value V(s)
# ------------------------------------------------------------
class Critic(nn.Module):
    def __init__(self, state_dim, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.v(x)


# ------------------------------------------------------------
# A2C Agent
# ------------------------------------------------------------
class A2CAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        entropy_coef=0.01,
        value_coef=0.5,
        max_action=1.0,
    ):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_action = max_action

        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

    def select_action(self, state):
        state_t = torch.FloatTensor(state).to(device).unsqueeze(0)
        action, log_prob, entropy = self.actor.sample(state_t)
        action = action.cpu().numpy()[0] # original
        #action = action.cpu().detach().numpy()[0]
        action = np.clip(action, -self.max_action, self.max_action)
        return action, log_prob, entropy

    def update(self, state, reward, next_state, done, log_prob, entropy):
        state_t = torch.FloatTensor(state).to(device).unsqueeze(0)
        next_state_t = torch.FloatTensor(next_state).to(device).unsqueeze(0)

        value      = self.critic(state_t).squeeze(0)
        next_value = self.critic(next_state_t).squeeze(0).detach()

        # ---- 1-step TD target ----
        td_target = reward + self.gamma * next_value * (1 - done)

        # Advantage = TD residual
        advantage = td_target - value

        # ---- Losses ----
        actor_loss = -(log_prob * advantage.detach()) - self.entropy_coef * entropy
        critic_loss = F.mse_loss(value, td_target.detach())

        loss = actor_loss + self.value_coef * critic_loss

        # ---- Gradient step ----
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
        }


# ------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------
def train_a2c(
    env_name="Pendulum-v1",
    episodes=300,
    max_steps=200,
):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = A2CAgent(state_dim, action_dim, max_action=max_action)

    for ep in range(episodes):
        state = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            action, log_prob, entropy = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            metrics = agent.update(
                state, reward, next_state, float(done), log_prob, entropy
            )

            state = next_state
            if done:
                break

        if ep % 10 == 0:
            print(
                f"Episode {ep}, Reward = {episode_reward:.2f}, "
                f"A-Loss = {metrics['actor_loss']:.3f}, "
                f"C-Loss = {metrics['critic_loss']:.3f}"
            )

    return agent


# ------------------------------------------------------------
# Example: run training
# ------------------------------------------------------------
if __name__ == "__main__":
    train_a2c(env_name="Pendulum-v1", episodes=200, max_steps=200)