import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ================================================================
#   Actor network: produces Gaussian actions (independent)
# ================================================================
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        x = self.net(obs)
        mu = self.mu(x)
        std = torch.exp(self.log_std)
        return mu, std

    def act(self, obs):
        mu, std = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        return action, logp


# ================================================================
#   Centralized Critic: takes JOINT observation
# ================================================================
class CentralCritic(nn.Module):
    def __init__(self, joint_obs_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, joint_obs):
        return self.net(joint_obs)


# ================================================================
#             Multi-Agent A2C with Centralized Critic
# ================================================================
class MAAC_A2C:
    def __init__(self, num_agents, obs_dim, act_dim, gamma=0.99, lr=3e-4):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.gamma = gamma

        # Independent actors
        self.actors = [
            Actor(obs_dim, act_dim) for _ in range(num_agents)
        ]
        self.actor_optim = [
            optim.Adam(a.parameters(), lr=lr) for a in self.actors
        ]

        # Centralized critic
        self.critic = CentralCritic(joint_obs_dim=num_agents * obs_dim)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

    # ------------------------------------------------------------
    def act(self, obs_batch):
        """
        obs_batch: tensor [num_agents, obs_dim]
        Returns:
            actions [num_agents, act_dim]
            logps   [num_agents]
        """
        actions, logps = [], []
        for i, actor in enumerate(self.actors):
            a, logp = actor.act(obs_batch[i])
            actions.append(a)
            logps.append(logp)
        return torch.stack(actions), torch.stack(logps)

    # ------------------------------------------------------------
    def update(self, obs, actions, logps, rewards, next_obs, dones):
        """
        obs:      [num_agents, obs_dim]
        next_obs: [num_agents, obs_dim]
        rewards:  [num_agents]
        dones:    [num_agents]
        """

        # Flatten joint obs
        joint_obs = obs.reshape(-1)
        joint_next_obs = next_obs.reshape(-1)

        # Critic values
        V = self.critic(joint_obs)
        with torch.no_grad():
            V_next = self.critic(joint_next_obs)
            td_target = rewards.sum() + self.gamma * (1 - dones.max()) * V_next
            advantage = td_target - V

        # ---------------- Critic update ----------------
        critic_loss = advantage.pow(2)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # ---------------- Actor updates ----------------
        # Each actor gets the same centralized advantage
        for i, actor in enumerate(self.actors):
            policy_loss = -(logps[i] * advantage.detach())

            self.actor_optim[i].zero_grad()
            policy_loss.backward()
            self.actor_optim[i].step()


# ================================================================
# Example training loop using dummy environment
# ================================================================
if __name__ == "__main__":
    num_agents = 3
    obs_dim = 4
    act_dim = 2

    class DummyMAEnv:
        def reset(self):
            return np.random.randn(num_agents, obs_dim).astype(np.float32)

        def step(self, actions):
            next_state = np.random.randn(num_agents, obs_dim).astype(np.float32)
            reward = np.random.randn(num_agents).astype(np.float32)
            done = np.random.choice([0, 1], size=num_agents, p=[0.95, 0.05])
            return next_state, reward, done

    env = DummyMAEnv()
    agent = MAAC_A2C(num_agents, obs_dim, act_dim)

    obs = torch.tensor(env.reset(), dtype=torch.float32)

    for t in range(20000):
        actions, logps = agent.act(obs)

        next_obs_np, rewards_np, dones_np = env.step(actions.detach().numpy())

        rewards = torch.tensor(rewards_np, dtype=torch.float32)
        dones = torch.tensor(dones_np, dtype=torch.float32)
        next_obs = torch.tensor(next_obs_np, dtype=torch.float32)

        agent.update(obs, actions, logps, rewards, next_obs, dones)
        obs = next_obs