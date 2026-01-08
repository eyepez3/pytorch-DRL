import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from torchviz import make_dot


LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0
EPS = 1e-6

class DummyEnv:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def reset(self):
        return np.random.randn(self.state_dim).astype(np.float32)

    def step(self, action):
        # Reward encourages action near zero
        reward = -np.sum(action**2)
        next_state = np.random.randn(self.state_dim).astype(np.float32)
        done = False
        return next_state, reward, done

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        self.action_dim = action_dim

        # ---------- Shared trunk ----------
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )

        # ---------- Actor heads ----------
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        # ---------- Critic ----------
        self.value_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

        # Extra safety
        nn.init.constant_(self.log_std_head.bias, -0.5)

    def forward(self, obs):
        x = self.shared(obs)

        # ----- Actor -----
        raw_mu = self.mu_head(x)
        mu = torch.tanh(raw_mu)                # bounded mean

        raw_log_std = self.log_std_head(x)
        log_std = torch.clamp(raw_log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        dist = Normal(mu, std)

        # ----- Critic -----
        value = self.value_head(x).squeeze(-1)

        return dist, value, mu, std
    
    def getActorCritic(self, obs):
        dist, value, mu, std = self.forward(obs)

        # Reparameterized sample
        z = dist.rsample()
        action = torch.tanh(z)

        # ---- Correct tanh log-prob ----
        log_prob = dist.log_prob(z)
        log_prob -= torch.log(1 - action.pow(2) + EPS)
        log_prob = log_prob.sum(-1)

        # Optional: entropy (use in loss)
        entropy = dist.entropy().sum(-1)

        return action, log_prob, value, mu, std, entropy
    
class a2c_model(object):
    def __init__(self,state_dim,action_dim,num_agents=2,lr=4e-3,hidden_dim=128, \
                 model_path='./models',device='cpu',gamma=0.99,lamda=0.95):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.model_path = model_path
        self.device = device
        self.num_agents = num_agents
        self.gamma = gamma
        self.lr = lr
        self.lamda = lamda
        self.gae = 0
        self.epsilon = 1e-6
        self.render_graphs = True

        self.agents = []
        self.optimizers = []
        self.stop_update = []
        
        for _ in range(num_agents):
            net = ActorCritic(state_dim, action_dim,hidden_dim)
            self.agents.append(net)
            self.optimizers.append(
                optim.Adam(net.parameters(), lr=lr)
            )
            self.stop_update.append(False)

    def reset(self):
        self.next_state_list = []
        self.action_list = []
        self.mu_list = []
        self.sigma_list = []
        self.value_list = []
        self.target_list = []
        self.reward_list = []
        self.log_prob_list = []
        self.a_loss_list = []
        self.a_loss_mean_list = []
        self.c_loss_list = []
        self.done_list = []
        self.mask_list = []
        self.entropy = 0
        self.gae = np.array([0.0],dtype=np.float32)
        self.next_value = np.array([[0.0]],dtype=np.float32)
        self.stop_update = []
        for i in range(self.num_agents):
            self.stop_update.append(False)
    
    def getActionValue(self,state_in,id):

        agent = self.agents[id]
        #state_t = torch.FloatTensor(state_np).to(self.device).unsqueeze[0]
        if isinstance(state_in, np.ndarray):
            state_in = torch.tensor(state_in, dtype=torch.float32)
        actions, logprob_t, values, mu, std = agent.getActorCritic(state_in)

        # create computational graph
        if (self.render_graphs):
            dot = make_dot(logprob_t, params=dict(agent.named_parameters()))
            dot.render("logp1_graph_"+str(id), format="pdf")
            dot = make_dot(values, params=dict(agent.named_parameters()))
            dot.render("critic_graph_"+str(id), format="pdf")

        # convert tensors to numpy and detach from graph
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().numpy()
        if isinstance(mu, torch.Tensor):
            mu = mu.detach().numpy()
        if isinstance(std, torch.Tensor):
            std = std.detach().numpy()
        if isinstance(values, torch.Tensor):
            values = values.detach().numpy()

        return (actions,logprob_t,values,mu,std)

    def update(self, curr_obs, next_obs, actions, rewards, dones):
        """
        Clean A2C update with entropy regularization and diagnostics.

        Inputs are lists (len = num_agents):
        curr_obs[i]  : state_t
        next_obs[i]  : state_{t+1}
        actions[i]   : action_t (already tanh-squashed)
        rewards[i]   : reward_t (float)
        dones[i]     : done flag (0/1)
        """

        entropy_coef = 0.01
        value_coef = 0.5
        max_grad_norm = 0.5

        diagnostics = {
            "mu_mean": [],
            "mu_std": [],
            "std_mean": [],
            "std_min": [],
            "std_max": [],
            "entropy": [],
            "policy_loss": [],
            "value_loss": [],
        }

        for i, agent in enumerate(self.agents):

            if self.stop_update[i]:
                continue

            # --- tensors ---
            obs_t = torch.as_tensor(curr_obs[i], dtype=torch.float32)
            next_obs_t = torch.as_tensor(next_obs[i], dtype=torch.float32)
            action_t = torch.as_tensor(actions[i], dtype=torch.float32)

            reward = rewards[i]
            done = dones[i]

            # ---------- Critic target ----------
            with torch.no_grad():
                _, next_value, _, _ = agent.forward(next_obs_t)
                td_target = reward + self.gamma * (1 - done) * next_value

            # ---------- Forward current ----------
            dist, value, mu, std = agent.forward(obs_t)

            # ---------- Log prob (inverse tanh) ----------
            eps = 1e-6
            atanh_action = torch.atanh(torch.clamp(action_t, -1 + eps, 1 - eps))
            log_prob = dist.log_prob(atanh_action)
            log_prob -= torch.log(1 - action_t.pow(2) + eps)
            log_prob = log_prob.sum(-1)

            # ---------- Advantage ----------
            advantage = (td_target - value).detach()

            # ---------- Losses ----------
            policy_loss = -(log_prob * advantage).mean()
            value_loss = value_coef * (td_target - value).pow(2).mean()
            entropy = dist.entropy().sum(-1).mean()

            loss = policy_loss + value_loss - entropy_coef * entropy

            # ---------- Backprop ----------
            self.optimizers[i].zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            self.optimizers[i].step()

            # ---------- Diagnostics ----------
            diagnostics["mu_mean"].append(mu.mean().item())
            diagnostics["mu_std"].append(mu.std().item())
            diagnostics["std_mean"].append(std.mean().item())
            diagnostics["std_min"].append(std.min().item())
            diagnostics["std_max"].append(std.max().item())
            diagnostics["entropy"].append(entropy.item())
            diagnostics["policy_loss"].append(policy_loss.item())
            diagnostics["value_loss"].append(value_loss.item())

            if done:
                self.stop_update[i] = True

            if (self.render_graphs):
                    dot = make_dot(loss, params=dict(agent.named_parameters()))
                    dot.render("loss_graph"+str(i), format="pdf")
                    if (i == 0):
                        self.render_graphs = False

        return diagnostics
    
    def print_diagnostics(self, diagnostics, step):
        if not diagnostics["mu_mean"]:
            return

        print(
            f"[Step {step}] "
            f"μ: {np.mean(diagnostics['mu_mean']):+.3f} ±{np.mean(diagnostics['mu_std']):.3f} | "
            f"σ: mean={np.mean(diagnostics['std_mean']):.3f} "
            f"min={np.min(diagnostics['std_min']):.3f} "
            f"max={np.max(diagnostics['std_max']):.3f} | "
            f"H={np.mean(diagnostics['entropy']):.3f}"
        )

def run_stability_test():
    state_dim = 8
    action_dim = 2
    num_agents = 2

    envs = [DummyEnv(state_dim, action_dim) for _ in range(num_agents)]
    
    model = a2c_model(
        state_dim=state_dim,
        action_dim=action_dim,
        num_agents=num_agents,
        lr=3e-4
    )

    states = [env.reset() for env in envs]

    for step in range(1, 501):

        actions = []
        values = []

        #action, log_prob, value, mu, std, entropy
        for i in range(num_agents):
            a, _, _, mu, std,_ = model.agents[i].getActorCritic(
                torch.tensor(states[i], dtype=torch.float32)
            )
            actions.append(a.detach().numpy())

        next_states, rewards, dones = [], [], []

        for i, env in enumerate(envs):
            ns, r, d = env.step(actions[i])
            next_states.append(ns)
            rewards.append(r)
            dones.append(d)

        diagnostics = model.update(
            curr_obs=states,
            next_obs=next_states,
            actions=actions,
            rewards=rewards,
            dones=dones
        )

        if step % 25 == 0:
            model.print_diagnostics(diagnostics, step)

        states = next_states

# ============================================================
if __name__ == "__main__":
    run_stability_test()