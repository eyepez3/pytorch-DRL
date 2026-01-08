import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# =========================
# Hyperparameters
# =========================
STATE_DIM = 8
ACTION_DIM = 2
NUM_AGENTS = 3
HIDDEN_DIM = 128
LR = 3e-4
GAMMA = 0.99
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
EPS = 1e-6

LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0

DEVICE = "cpu"

# =========================
# Dummy Environment
# =========================
class DummyEnv:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def reset(self):
        return np.random.randn(self.state_dim).astype(np.float32)

    def step(self, action):
        # Encourage small actions
        reward = -np.sum(action ** 2)
        next_state = np.random.randn(self.state_dim).astype(np.float32)
        done = False
        return next_state, reward, done

# =========================
# Shared Actor-Critic with Agent ID Embeddings
# =========================
class SharedActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents, hidden_dim):
        super().__init__()

        self.agent_embed = nn.Embedding(num_agents, hidden_dim)

        self.shared = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )

        # Actor heads
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        # Critic
        self.value_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)
        nn.init.constant_(self.log_std_head.bias, -0.5)

    def forward(self, obs, agent_id):
        """
        obs: (state_dim,)
        agent_id: scalar long tensor
        """
        agent_emb = self.agent_embed(agent_id)
        x = torch.cat([obs, agent_emb], dim=-1)

        x = self.shared(x)

        # Actor
        mu = torch.tanh(self.mu_head(x))
        log_std = torch.clamp(self.log_std_head(x), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = Normal(mu, std)

        # Critic
        value = self.value_head(x).squeeze(-1)

        return dist, value, mu, std

    def act(self, obs, agent_id):
        dist, value, mu, std = self.forward(obs, agent_id)

        z = dist.rsample()
        action = torch.tanh(z)

        log_prob = dist.log_prob(z)
        log_prob -= torch.log(1 - action.pow(2) + EPS)
        log_prob = log_prob.sum(-1)

        entropy = dist.entropy().sum(-1)

        return action, log_prob, value, mu, std, entropy

# =========================
# Shared A2C Trainer
# =========================
class SharedA2C:
    def __init__(self):
        self.model = SharedActorCritic(
            STATE_DIM, ACTION_DIM, NUM_AGENTS, HIDDEN_DIM
        ).to(DEVICE)

        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

    def update(self, batch):
        """
        batch: list of dicts, one per agent
        """
        policy_losses = []
        value_losses = []
        entropies = []

        for data in batch:
            obs = data["obs"]
            next_obs = data["next_obs"]
            action = data["action"]
            reward = data["reward"]
            done = data["done"]
            agent_id = data["agent_id"]

            # Critic target
            with torch.no_grad():
                _, next_value, _, _ = self.model.forward(next_obs, agent_id)
                td_target = reward + GAMMA * (1 - done) * next_value

            # Current evaluation
            dist, value, mu, std = self.model.forward(obs, agent_id)

            atanh_action = torch.atanh(torch.clamp(action, -1 + EPS, 1 - EPS))
            log_prob = dist.log_prob(atanh_action)
            log_prob -= torch.log(1 - action.pow(2) + EPS)
            log_prob = log_prob.sum(-1)

            advantage = (td_target - value).detach()

            policy_losses.append(-(log_prob * advantage))
            value_losses.append((td_target - value).pow(2))
            entropies.append(dist.entropy().sum(-1))

        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        entropy = torch.stack(entropies).mean()

        loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }

# =========================
# Training Loop
# =========================
def run_shared_a2c():
    envs = [DummyEnv(STATE_DIM, ACTION_DIM) for _ in range(NUM_AGENTS)]
    agent = SharedA2C()

    states = [env.reset() for env in envs]

    for step in range(1, 501):
        batch = []

        for i in range(NUM_AGENTS):
            obs = torch.tensor(states[i], dtype=torch.float32)
            agent_id = torch.tensor(i, dtype=torch.long)

            action, logp, value, mu, std, entropy = agent.model.act(obs, agent_id)
            next_state, reward, done = envs[i].step(action.detach().numpy())

            batch.append({
                "obs": obs,
                "next_obs": torch.tensor(next_state, dtype=torch.float32),
                "action": action.detach(),
                "reward": torch.tensor(reward, dtype=torch.float32),
                "done": torch.tensor(done, dtype=torch.float32),
                "agent_id": agent_id
            })

            states[i] = next_state

        metrics = agent.update(batch)

        if step % 25 == 0:
            print(
                f"[Step {step}] "
                f"Policy={metrics['policy_loss']:.3f} | "
                f"Value={metrics['value_loss']:.3f} | "
                f"Entropy={metrics['entropy']:.3f}"
            )

# =========================
# Run
# =========================
if __name__ == "__main__":
    run_shared_a2c()