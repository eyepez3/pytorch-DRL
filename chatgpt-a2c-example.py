"""
A2C (Advantage Actor-Critic) for continuous action spaces using PyTorch.

Dependencies:
    pip install torch gym numpy

This version:
 - Uses a Gaussian policy: actor outputs mean + log_std
 - Entropy regularization
 - Computes advantages using a 1-step bootstrapped target
 - Works on envs like Pendulum-v1, MountainCarContinuous-v0, etc.

Author: ChatGPT
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device type = :',device)


# -------------------------------------------------------
# Actor-Critic Networks
# -------------------------------------------------------
class Actor(nn.Module):
    """Gaussian policy for continuous action spaces."""
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.mean_head = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean_head(x)
        std = torch.exp(self.log_std)   # softplus alternative also possible
        return mean, std

    def get_action(self, state):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std) # creates handle to dist object
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        return action, log_prob, dist

class Critic(nn.Module):
    """State-value network."""
    def __init__(self, state_dim, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.v_head = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v = self.v_head(x)
        return v


# -------------------------------------------------------
# A2C Agent
# -------------------------------------------------------
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

        # networks
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)

        # shared optimizer
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

    def select_action(self, state):
        state_np = np.array(state)
        state_t = torch.FloatTensor(state_np).to(device).unsqueeze(0)
        action, log_prob, _ = self.actor.get_action(state_t)
        #move to cpu, detach from the computational graph, convert to numpy array
        action = action.cpu().detach().numpy()[0]  
        return np.clip(action, -self.max_action, self.max_action), log_prob

    def compute_loss(self, trajectory):
        """
        trajectory = list of dictionaries with fields:
          state, action, reward, next_state, done, log_prob
        """
        states      = torch.FloatTensor([t["state"] for t in trajectory]).to(device)
        actions     = torch.FloatTensor([t["action"] for t in trajectory]).to(device)
        rewards     = torch.FloatTensor([t["reward"] for t in trajectory]).to(device)
        next_states = torch.FloatTensor([t["next_state"] for t in trajectory]).to(device)
        dones       = torch.FloatTensor([t["done"] for t in trajectory]).to(device)
        old_log_probs = torch.stack([t["log_prob"] for t in trajectory]).detach()

        # Critic values
        values = self.critic(states).squeeze(-1)
        next_values = self.critic(next_states).squeeze(-1).detach()

        # 1-step TD target
        td_target = rewards + self.gamma * next_values * (1 - dones)
        advantage = td_target - values

        # Compute log_probs of actions taken
        means, stds = self.actor.forward(states)
        dist = torch.distributions.Normal(means, stds)
        log_probs = dist.log_prob(actions).sum(axis=-1)

        # Actor loss = policy gradient
        actor_loss = -(advantage.detach() * log_probs).mean()

        # Critic loss = value function regression
        critic_loss = F.mse_loss(values, td_target.detach())

        # Entropy (encourages exploration)
        entropy = dist.entropy().sum(axis=-1).mean()

        total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

        return total_loss, actor_loss.item(), critic_loss.item(), entropy.item()

    def update(self, trajectory):
        loss, a_loss, c_loss, ent = self.compute_loss(trajectory)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "actor_loss": a_loss,
            "critic_loss": c_loss,
            "entropy": ent,
        }


# -------------------------------------------------------
# Training Loop
# -------------------------------------------------------
def train_a2c(
    env_name="Pendulum-v1",
    episodes=300,
    steps_per_update=200,
):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = A2CAgent(state_dim, action_dim, max_action=max_action)

    returns = []

    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0

        trajectory = []

        for step in range(steps_per_update):
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            trajectory.append(
                {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "done": float(done),
                    "log_prob": log_prob,
                }
            )

            state = next_state
            ep_reward += reward

            if done:
                state = env.reset()

        # Update agent after collecting rollout
        metrics = agent.update(trajectory)
        returns.append(ep_reward)

        if ep % 10 == 0:
            print(
                f"Episode {ep}, Return={ep_reward:.2f}, "
                f"Loss={metrics['loss']:.3f}, "
                f"A={metrics['actor_loss']:.3f}, "
                f"C={metrics['critic_loss']:.3f}, "
                f"Ent={metrics['entropy']:.3f}"
            )

    return agent


# -------------------------------------------------------
# Run example
# -------------------------------------------------------
if __name__ == "__main__":
    agent = train_a2c("Pendulum-v1", episodes=200, steps_per_update=200)