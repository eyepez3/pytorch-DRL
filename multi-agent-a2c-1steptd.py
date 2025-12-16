import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ================================================================
#   Gaussian Policy + Value network
# ================================================================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        
        # Shared base
        self.base = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        
        # Mean of action distribution
        self.mu_head = nn.Linear(hidden, act_dim)
        
        # Log std (learnable)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        
        # Value head
        self.v_head = nn.Linear(hidden, 1)

    def forward(self, obs):
        x = self.base(obs)
        mu = self.mu_head(x)
        std = torch.exp(self.log_std)
        value = self.v_head(x)
        return mu, std, value

    def act(self, obs):
        mu, std, value = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value


# ================================================================
#   Multi-Agent A2C
# ================================================================
class MultiAgentA2C:
    def __init__(self, num_agents, obs_dim, act_dim, gamma=0.99, lr=3e-4):
        self.num_agents = num_agents
        self.gamma = gamma
        
        self.agents = []
        self.optimizers = []
        
        for _ in range(num_agents):
            net = ActorCritic(obs_dim, act_dim)
            self.agents.append(net)
            self.optimizers.append(
                optim.Adam(net.parameters(), lr=lr)
            )

    def step(self, obs_batch):
        """
        obs_batch: tensor shape [num_agents, obs_dim]
        """
        actions = []
        log_probs = []
        values = []
        
        for i, agent in enumerate(self.agents):
            a, logp, v = agent.act(obs_batch[i])
            actions.append(a)
            log_probs.append(logp)
            values.append(v)

        return (
            torch.stack(actions),
            torch.stack(log_probs),
            torch.stack(values)
        )

    def update(self, obs, actions, log_probs, values, rewards, next_obs, dones):
        """
        obs:      [num_agents, obs_dim]
        next_obs: [num_agents, obs_dim]
        rewards:  [num_agents]
        dones:    [num_agents]
        """

        for i, agent in enumerate(self.agents):
            # Compute TD target
            with torch.no_grad():
                _, _, next_v = agent.forward(next_obs[i])
                td_target = rewards[i] + self.gamma * (1 - dones[i]) * next_v.item()
            
            # Advantage
            advantage = td_target - values[i]

            # Policy loss
            policy_loss = -(log_probs[i] * advantage.detach())

            # Value loss
            value_loss = advantage.pow(2)

            # Total loss
            loss = policy_loss + value_loss

            # Backprop
            self.optimizers[i].zero_grad()
            loss.backward()
            self.optimizers[i].step()

# ================================================================
# Example training loop (requires custom multi-agent environment)
# ================================================================
if __name__ == "__main__":
    num_agents = 3
    obs_dim = 5
    act_dim = 2
    
    # Example dummy environment
    class DummyMAEnv:
        def reset(self):
            return np.random.randn(num_agents, obs_dim).astype(np.float32)
        
        def step(self, actions):
            next_state = np.random.randn(num_agents, obs_dim).astype(np.float32)
            reward = np.random.randn(num_agents).astype(np.float32)
            done = np.random.choice([0, 1], size=num_agents, p=[0.95, 0.05])
            return next_state, reward, done
    
    env = DummyMAEnv()
    agent = MultiAgentA2C(num_agents, obs_dim, act_dim)
    
    obs = torch.tensor(env.reset())

    for step in range(10000):
        actions, logps, values = agent.step(obs)

        # Convert to numpy for env
        next_obs_np, rewards_np, dones_np = env.step(actions.detach().numpy())
        
        rewards = torch.tensor(rewards_np, dtype=torch.float32)
        dones = torch.tensor(dones_np, dtype=torch.float32)
        next_obs = torch.tensor(next_obs_np, dtype=torch.float32)

        # A2C update
        agent.update(obs, actions, logps, values, rewards, next_obs, dones)

        obs = next_obs