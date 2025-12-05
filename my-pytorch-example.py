"""
Soft Actor-Critic (SAC) implementation in PyTorch.

Dependencies:
    pip install torch gym numpy

Notes:
 - The code uses a tanh-squashed Gaussian policy and applies the appropriate
   log-probability correction for the tanh transform.
 - Automatic entropy tuning is implemented.
 - Uses twin Q-networks and target networks.
 - Minimalistic logger: prints episode reward and loss occasionally.

Adapt for your environment and hyperparameters as desired.
"""

import math
import random
from collections import deque, namedtuple
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ----------------------------
# Utilities & Replay Buffer
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

# ----------------------------
# Neural Network Models
# ----------------------------
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=(256,256), activation=F.relu):
        super().__init__()
        sizes = [input_dim] + list(hidden_sizes) + [output_dim]
        layers = []
        for i in range(len(sizes)-2):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.net = nn.Sequential(*layers)
        self.activation = activation

    def forward(self, x):
        return self.net(x)

# Q-network: outputs scalar Q(s,a)
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(256,256)):
        super().__init__()
        self.net = MLP(state_dim + action_dim, 1, hidden_sizes)
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        q = self.net(x)
        return q.squeeze(-1)  # shape: (batch,)

# Gaussian Policy with tanh squashing
class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(256,256), log_std_min=-20, log_std_max=2):
        super().__init__()
        self.net = MLP(state_dim, action_dim*2, hidden_sizes)  # outputs mean and log_std concatenated
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        # returns mean, log_std (unsquashed)
        params = self.net(state)
        mean, log_std = torch.chunk(params, 2, dim=-1)
        log_std = torch.tanh(log_std)  # keep it bounded then scale
        log_std = self.log_std_min + 0.5*(self.log_std_max - self.log_std_min)*(log_std + 1)
        return mean, log_std

    def sample(self, state, epsilon=1e-6):
        """
        Returns:
          action: squashed action in (-1,1)
          log_prob: log probability of the action (with tanh correction)
          pre_tanh: action before tanh (useful for debug)
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)

        normal = torch.distributions.Normal(mean, std)
        # reparameterize
        z = normal.rsample()  # pre-squash action
        action = torch.tanh(z)

        # log prob
        log_prob = normal.log_prob(z).sum(dim=-1)
        # correction for tanh squashing
        log_prob -= torch.sum(torch.log(1 - action.pow(2) + epsilon), dim=-1)
        return action, log_prob, torch.tanh(mean)

    def deterministic(self, state):
        mean, _ = self.forward(state)
        return torch.tanh(mean)

# ----------------------------
# SAC Agent
# ----------------------------
class SACAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_sizes=(256,256),
        replay_size=1_000_00,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,                # initial alpha; if automatic tuning enabled, this is initial value
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        automatic_entropy_tuning=True,
        target_entropy=None,
        batch_size=256,
        device=device,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.action_dim = action_dim

        # networks
        self.actor = GaussianPolicy(state_dim, action_dim, hidden_sizes).to(self.device)
        self.critic1 = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.critic2 = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.critic1_target = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.critic2_target = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # entropy tuning
        self.automatic_entropy_tuning = automatic_entropy_tuning
        if target_entropy is None:
            self.target_entropy = -action_dim  # heuristic from paper
        else:
            self.target_entropy = target_entropy

        if self.automatic_entropy_tuning:
            # we optimize log_alpha to ensure alpha stays positive
            self.log_alpha = torch.tensor(math.log(alpha), requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha
            self.log_alpha = None

        # replay buffer
        self.replay_buffer = ReplayBuffer(replay_size)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if evaluate:
            with torch.no_grad():
                action = self.actor.deterministic(state)
                return action.cpu().numpy()[0]
        else:
            with torch.no_grad():
                action, _, _ = self.actor.sample(state)
                return action.cpu().numpy()[0]

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None  # not enough samples yet

        transitions = self.replay_buffer.sample(self.batch_size)
        state = torch.FloatTensor(np.array(transitions.state)).to(self.device)
        action = torch.FloatTensor(np.array(transitions.action)).to(self.device)
        reward = torch.FloatTensor(np.array(transitions.reward)).to(self.device).unsqueeze(-1)
        next_state = torch.FloatTensor(np.array(transitions.next_state)).to(self.device)
        done = torch.FloatTensor(np.array(transitions.done)).to(self.device).unsqueeze(-1)

        # ----------------------------
        # Update critic (Q) networks
        # ----------------------------
        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor.sample(next_state)
            # next_q from target critics, take min
            q1_next = self.critic1_target(next_state, next_action)
            q2_next = self.critic2_target(next_state, next_action)
            q_next = torch.min(q1_next, q2_next).unsqueeze(-1)  # shape (batch,1)
            target_q = reward + (1 - done) * self.gamma * (q_next - self.alpha * next_log_pi.unsqueeze(-1))

        # current Q estimates
        current_q1 = self.critic1(state, action).unsqueeze(-1)
        current_q2 = self.critic2(state, action).unsqueeze(-1)

        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # ----------------------------
        # Update actor (policy)
        # ----------------------------
        new_action, log_pi, _ = self.actor.sample(state)
        q1_new = self.critic1(state, new_action)
        q2_new = self.critic2(state, new_action)
        q_new = torch.min(q1_new, q2_new)

        policy_loss = (self.alpha * log_pi - q_new).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # ----------------------------
        # Update alpha (entropy temperature)
        # ----------------------------
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        else:
            alpha_loss = torch.tensor(0.)

        # ----------------------------
        # Soft update target networks
        # ----------------------------
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha_loss': alpha_loss.item() if isinstance(alpha_loss, torch.Tensor) else float(alpha_loss),
            'alpha': self.alpha
        }

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

# ----------------------------
# Training loop example
# ----------------------------
def train_sac(
    env_name='Pendulum-v1',
    seed=0,
    episodes=500,
    max_steps=200,
    start_steps=1000,
    update_after=1000,
    update_every=1,
    batch_size=256,
    log_interval=10
):
    # create env
    env = gym.make(env_name)
    #env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    # action space assumed to be Box
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        batch_size=batch_size,
        automatic_entropy_tuning=True,
    )

    total_steps = 0
    episode_rewards = []

    for ep in range(1, episodes+1):
        state = env.reset()
        ep_reward = 0.0
        for step in range(max_steps):
            if total_steps < start_steps:
                # sample random action for initial exploration
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, evaluate=False)
                # action returned is in (-1,1); scale to env action range
                action = np.clip(action, -1, 1) * max_action

            next_state, reward, done, info = env.step(action)
            ep_reward += reward

            agent.store_transition(state, action / max_action, reward, next_state, float(done))  # store normalized action (-1..1)
            state = next_state
            total_steps += 1

            # update agent
            if total_steps >= update_after and total_steps % update_every == 0:
                _loss_info = agent.update()

            if done or step == max_steps-1:
                episode_rewards.append(ep_reward)
                break

        if ep % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            print(f"EP {ep}\tAvgReward(last {log_interval}): {avg_reward:.2f}\tTotalSteps: {total_steps}\tAlpha: {agent.alpha:.4f}")

    return agent, env

# ----------------------------
# Example run (uncomment to run)
# ----------------------------
if __name__ == "__main__":
    # Quick example: train on Pendulum (slow if GPU disabled)
    trained_agent, environment = train_sac(
        env_name='Pendulum-v1',
        seed=42,
        episodes=200,
        max_steps=200,
        start_steps=1000,
        update_after=1000,
        update_every=1,
        batch_size=128,
        log_interval=10
    )