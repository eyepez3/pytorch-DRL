import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from torchviz import make_dot
import matplotlib.pyplot as plt

def add_angles_wrap_2pi(a1, a2):
    """
    Add two angles and wrap the result to [-2π, 2π].

    Parameters
    ----------
    a1, a2 : float
        Angles in radians

    Returns
    -------
    float
        Angle wrapped to [-2π, 2π]
    """
    angle = a1 + a2
    two_pi = 2 * np.pi

    # Wrap to [-2π, 2π]
    angle = (angle + two_pi) % (4 * np.pi) - two_pi

    return angle

def wrapAngle_pi(angle):
    angle2 = np.copy(angle)
    sign = np.sign(angle)
    if np.abs(angle2) > np.pi:
        print('|angle| > pi')
        angle2 = sign*2*np.pi - angle2
    angle2 = np.arctan2(np.sin(angle2), np.cos(angle2))
    return(angle2)

#def angle_diff_wrap_2pi(a1, a2):
def getAngle_2pi(a1, a2):
    """
    Signed angle difference (a2 - a1), wrapped to [-2π, 2π].

    Parameters
    ----------
    a1, a2 : float or np.ndarray
        Angles in radians

    Returns
    -------
    diff : float or np.ndarray
        Wrapped signed angle difference
    """
    diff = a2 - a1
    two_pi = 2 * np.pi

    # Wrap into [-2π, 2π]
    diff = (diff + two_pi) % (4 * np.pi) - two_pi

    return diff

def getAngle1(a1, a2):
    """
    Compute unwrapped signed angle difference (a2 - a1) in radians.
    Result is in (-pi, pi].

    Parameters
    ----------
    a1, a2 : float or np.ndarray
        Angles in radians
    Positive  -> Counter-clockwise (CCW)
    Negative  -> Clockwise (CW)
    Zero      -> No rotation

    Returns
    -------
    diff : float or np.ndarray
        Signed angle difference
    """
    diff = np.arctan2(np.sin(a2 - a1), np.cos(a2 - a1))
    if diff > 0:
        direction = "CCW"
    elif diff < 0:
        direction = "CW"
    else:
        direction = "NONE"
    print(f"Angle difference (deg): {np.rad2deg(diff):.2f}")
    print(f"Direction: {direction}")

    return diff

def getAngle2(dir1,dir2,ref_norm=None):
    
    # assume unity vector with direction (angle) as input
    # determines angle between velocity vectors
    # determines sign: pos=ccw=left,neg=cw=right
    v1 = np.array([np.cos(dir1),np.sin(dir1),0])
    v2 = np.array([np.cos(dir2),np.sin(dir2),0])
    
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)
    # avoid divide by zero
    if mag_v1 == 0 or mag_v2 == 0:
        return 0.0
    dot_prod = np.dot(v1,v2)
    cross_prod = np.cross(v1,v2)
    if (ref_norm is None):
        norm_ref = cross_prod
    else:
        norm_ref = ref_norm
    cross_sign = cross_prod[2]
    angle_sign = 1.0
    if (cross_sign < 0):
        angle_sign = -1.0
    cos_angle = dot_prod/(mag_v1*mag_v2)
    cos_angle = np.clip(cos_angle,-1.0,1.0)
    angle_rad = np.arccos(cos_angle)*angle_sign
    return angle_rad

def getRelDistDir(posA, posB):
    # (direction values should be +/- pi) verified via matlab on 12/7/2025
    dX,dY = posB - posA
    range = np.linalg.norm(posB - posA)
    rel_dir = np.arctan2(dY,dX) # radians
    return(range,rel_dir)

def plot_trial_metrics(step_data, metrics, title="Training Metrics", normalize=False):
    """
    step_data : list or np.ndarray
        Step indices (e.g. [0, 1, 2, ..., T])
    metrics : dict
        { "name": list or np.ndarray of values }
    title : str
        Plot title
    normalize : bool
        If True, normalize each metric to [0, 1]
    """

    plt.figure(figsize=(10, 5))

    for name, values in metrics.items():
        values = np.asarray(values)

        if normalize:
            min_v, max_v = values.min(), values.max()
            if max_v > min_v:
                values = (values - min_v) / (max_v - min_v)

        plt.plot(step_data, values, label=name)

    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def create_figure(trial,metrics):
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, num=trial, figsize=(10, 2.5 * n), sharex=True)
    return(fig,axes)

def plot_trial_metrics_subplots(fig,axes,step_data, metrics, title="Training Metrics", normalize=False):
    #fig = plt.figure(trial)
    n = len(metrics)
    if n == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, metrics.items()):
        if normalize:
            min_v, max_v = values.min(), values.max()
            if max_v > min_v:
                values = (values - min_v) / (max_v - min_v)
        if isinstance(values, torch.Tensor):
            values = values.detach().numpy()
        ax.plot(step_data, values)
        ax.set_ylabel(name)
        ax.grid(True)

    axes[-1].set_xlabel("Step")
    fig.suptitle(title)
    plt.tight_layout()
    #plt.show()
    plt.pause(.01)

# --- 1. Define the Actor Network Class ---

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
#from myutils import flatten
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from torchviz import make_dot

#state_size = env.observation_space.shape[0]
#action_size = env.action_space.n

class ActorCritic(nn.Module):
    def __init__(self,state_dim,action_dim,hidden_dim=128,lr=4e-3,gamma=0.99):
        super(ActorCritic, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.var_bias = 1e-6
        self.lr = lr
        self.gamma = gamma

        # base network (policy)
        self.base = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor Mu network (policy)
        self.actor_mu = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        # Log std (learnable)
        #self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.actor_var = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim ),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softplus()
        )
        # critic output
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state_in):
        base_out = self.base(state_in)
        # Critic forward pass
        value = self.critic(base_out)

        # Actor forward pass
        #mu = self.actor_mu(base_out)
        mu = self.actor_mu(base_out)
        #std = torch.exp(self.log_std)
        std = self.actor_var(base_out) + 1e-6
        #std = torch.sqrt(var)
        
        return (mu,std,value)
    
    def getActionValue(self, state_in):
        mu, std, value = self.forward(state_in)
        mu = torch.clamp(mu, -0.999, 0.999)
        std = torch.clamp(std, 1e-6, 10.0)
        # NaN guard
        if torch.isnan(mu).any() or torch.isnan(std).any():
            raise ValueError(f"NaN detected: mu={mu}, std={std}")
        dist = torch.distributions.Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        #action_np = action.detach().numpy()
        #mu_np = mu.detach().numpy()
        #std_np = std.detach().numpy()
        return (action, log_prob, value, mu, std)

class a2c_model(object):
    def __init__(self,state_dim,action_dim,hidden_dim=128, \
                 model_path='./models',device='cpu', \
                 num_agents=2,gamma=0.99,lr=4e-3,lamda=0.95):
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
        
        for _ in range(num_agents):
            net = ActorCritic(state_dim, action_dim,hidden_dim,lr,gamma)
            self.agents.append(net)
            self.optimizers.append(
                optim.Adam(net.parameters(), lr=lr)
            )

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
        
    def printTrainableParams(self,id):
        agent = self.agents[id]
        for name, param in agent.named_parameters():
            if param.requires_grad:
                print(name, param.data)
    
    def transferParams(self,last_id,print_params):
        # transfer parameters from last surviving agent to other agents
        Sagent = self.agents[last_id]
        if (print_params):
            print('source %d agent params:' % (last_id))
            self.printTrainableParams(last_id)

        for i in range (self.num_agents):
            if (i != last_id):
                Tagent = self.agents[i]
                if (print_params):
                    print('target %d agent params before:' % (i))
                    self.printTrainableParams(i)
                Tagent.load_state_dict(Sagent.state_dict())
                if (print_params):
                    print('target %d agent params after:' % (i))
                    self.printTrainableParams(i)
    
    def getActionValue(self,state_t,id=0):

        agent = self.agents[id]
        #state_t = torch.FloatTensor(state_np).to(self.device).unsqueeze[0]
        if isinstance(state_t, np.ndarray):
            state_t = torch.tensor(state_t, dtype=torch.float32)
        actions, logprob_t, value_t, mu, std = agent.getActionValue(state_t)
        
        # create computational graph
        if (self.render_graphs):
            dot = make_dot(logprob_t, params=dict(agent.named_parameters()))
            dot.render("logp1_graph_"+str(id), format="pdf")
            dot = make_dot(value_t, params=dict(agent.named_parameters()))
            dot.render("value_t_graph_"+str(id), format="pdf")

        return (actions,logprob_t,value_t,mu,std)
    
    def update(self, log_probs_t, curr_obs_t, next_obs_t, rewards, dones):
        """
        Assumes:
        - log_probs_t: list (or iterable) of tensors (one per agent) possibly coming from the rollout.
        - values_t: list of tensors that were returned during the rollout (we won't backprop through them).
        - curr_obs, next_obs: lists/arrays of per-agent observations (numpy or tensors).
        - rewards, dones: lists/arrays per-agent scalars.
        """
        closs_list = []
        aloss_list = []
        tloss_list = []
        for i, agent in enumerate(self.agents):

            if (self.stop_update[i] == False):
                # --- prepare tensors for obs/next_obs ---
                if isinstance(curr_obs_t, np.ndarray):
                    curr_obs_t = torch.tensor(curr_obs_t, dtype=torch.float32)
                if isinstance(next_obs_t, np.ndarray):
                    next_obs_t = torch.tensor(next_obs_t, dtype=torch.float32)

                # ---- compute TD target using next state's value (no grad) ----
                with torch.no_grad():
                    _, _, next_value = agent.forward(next_obs_t)
                    # next_val is a 1-element tensor; convert to python float
                    td_target = rewards + self.gamma * (1 - dones) * float(next_value.item())

                # ---- recompute current value with a fresh forward pass (this will have grad) ----
                mu, std, curr_value = agent.forward(curr_obs_t)   # curr_value requires grad

                # ---- advantage: break graph by detaching curr_value here ----
                advantage = td_target - curr_value.detach()  # advantage has no grad
                

                # ---- policy loss: use the stored log_prob but DETACH it to avoid reusing the rollout graph ----
                # log_probs_t[i] might be a tensor shaped (1,) or scalar; ensure it's a tensor scalar
                if isinstance(log_probs_t, np.ndarray):
                    log_probs_t = torch.tensor(log_probs_t, dtype=torch.float32)
                #logp = logp.detach()
                # If logp has extra dims, reduce to scalar
                #logp = logp.squeeze()

                policy_loss = -(log_probs_t * advantage).sum()   # sum -> scalar (use mean() if you prefer)

                # ---- value loss: compare td_target to curr_value (this backprops into critic) ----
                # td_target -> tensor with same device/dtype as curr_value
                td_target_t = torch.tensor([td_target], dtype=curr_value.dtype)
                value_loss = (td_target_t - curr_value).pow(2).sum()  # scalar

                # ---- total loss and step ----
                loss = policy_loss + value_loss

                if (self.render_graphs):
                    dot = make_dot(log_probs_t, params=dict(agent.named_parameters()))
                    dot.render("logp2_graph_"+str(i), format="pdf")
                    dot = make_dot(advantage, params=dict(agent.named_parameters()))
                    dot.render("adv_graph_"+str(i), format="pdf")
                    dot = make_dot(policy_loss, params=dict(agent.named_parameters()))
                    dot.render("ploss_graph_"+str(i), format="pdf")
                    dot = make_dot(value_loss, params=dict(agent.named_parameters()))
                    dot.render("vloss_graph_"+str(i), format="pdf")
                    dot = make_dot(loss, params=dict(agent.named_parameters()))
                    dot.render("loss_graph_"+str(i), format="pdf")
                    if (i == 0):
                        self.render_graphs = False

                
                self.optimizers[i].zero_grad()
                loss.backward()
                self.optimizers[i].step()

                closs_np = value_loss.cpu().detach().numpy()
                aloss_np = policy_loss.cpu().detach().numpy()
                tloss_np = loss.cpu().detach().numpy()

                print('closs: %0.4e, aloss: %0.4e, tloss: %0.4e\n' % (closs_np,aloss_np,tloss_np))

                closs_list.append(closs_np)
                aloss_list.append(aloss_np)
                tloss_list.append(tloss_np)

                if (dones > 0):
                    self.stop_update[i] = True
                    print('stop updating agent %d' % (i))

        return(np.sum(closs_list),np.sum(aloss_list),np.sum(tloss_list))

# --- 2. Example Usage ---
if __name__ == '__main__':
    STATE_DIM = 2   # dir, goal_rdir
    ACTION_DIM = 1 # steer
    hidden_dim = 64 #128
    model_path = f'./models'
    device = 'cpu'
    num_minnows = 1
    
        
    # 1. Initialize the Actor
    mymodel = a2c_model(STATE_DIM,ACTION_DIM,hidden_dim,model_path,device,num_minnows)
    
    print(f"Actor Network Structure:\n{mymodel.agents[0]}")
    # 2. Create a dummy state input (batch size of 1)
    # Typically, you process batches of states from the environment or replay buffer
    dummy_state = torch.randn(1, STATE_DIM) 
    
    # --- A. Sampling for Training/Exploration ---
    print("\n--- A. Sampling for Training ---")
    
    # action: the squashed action in [-1, 1]
    # log_prob: the log likelihood of taking that action
    action, log_prob, value, mu, std,  = mymodel.getActionValue(dummy_state)
    
    print(f"Input State Shape: {dummy_state.shape}")
    print(f"Output Action (Sampled): {action.detach().numpy()}")
    print(f"Action Shape: {action.shape}")
    print(f"Mean (mu) Output: {mu.detach().numpy()}")
    print(f"Std Dev (std) Output: {std.detach().numpy()}")
    print(f"Log Probability: {log_prob.detach().item():.4f}")
    
    # Verification: Check that the sampled action is within the [-1, 1] bounds
    is_bounded = torch.all((action >= -1.0) & (action <= 1.0))
    print(f"Is action bounded [-1, 1]? {is_bounded.item()}")

    # --- B. Selecting Deterministic Action for Evaluation ---
    #print("\n--- B. Selecting Deterministic Action for Evaluation ---")
    
    # The deterministic action is simply the mean (mu)
    #deterministic_action = mymodel.agents[0].select_action(dummy_state)

    #print(f"Deterministic Action (Mu): {deterministic_action}")

    max_trials = 100
    max_steer = 15*np.pi/180

    dmax = 2*np.pi
    dmax = np.float32(dmax)

    max_steps = 1000
    trial_max_steps = max_steps

    trial_steps_list = []

    a1 = np.deg2rad(350)
    a2 = np.deg2rad(10)
    diff = getAngle_2pi(a1, a2)
    print("Angle diff (rad):", diff)
    print("Angle diff (deg):", np.rad2deg(diff))
    a1 = np.deg2rad(10)
    a2 = np.deg2rad(350)
    diff = getAngle_2pi(a1, a2)
    print("Angle diff (rad):", diff)
    print("Angle diff (deg):", np.rad2deg(diff))

    for trials in range(max_trials):

        # get relative posistion and direction
        pos = np.array([2.0,5.0])
        dir = 180*np.pi/180
        goal_pos = np.array([8.0,5.0])
        rdist,rdir = getRelDistDir(pos,goal_pos)
        rdir = np.float32(rdir)

        total_reward = 0
        done = False
        mymodel.reset()

        step_list = []
        dir_list = []
        curr_err_list = []
        new_dir_list = []
        steer_list = []
        value_list = []
        reward_list = []
        mu_list = []
        std_list = []
        closs_list = []
        ploss_list = []

        for steps in range(max_steps):

            
            # get desired direction
            curr_err = getAngle_2pi(dir,rdir)
            # get desired steer
            #desired_steer = np.clip(desired_dir,-max_steer,max_steer)

            state_in = np.array([dir/dmax,rdir/dmax])
            state_in_t = torch.tensor(state_in,dtype=torch.float32)
            # get model action
            action_t, log_prob, value, mu, std  = mymodel.getActionValue(state_in_t,0)
            value_np = value.detach().numpy()
            mu_np = mu.detach().numpy()
            std_np = std.detach().numpy()

            # apply action to steer direction
            action_np = action_t.detach().numpy()
            steer = action_np[0] * max_steer

            new_dir = np.float32(add_angles_wrap_2pi(dir,steer))
            new_dir2 = new_dir/dmax

            next_state = np.array([new_dir2,rdir/dmax],dtype='float32')
            next_state_t = torch.tensor(next_state,dtype=torch.float32)

            # determine reward
            new_err = getAngle_2pi(new_dir,rdir)
            
            # +1,-1 simple reward structure (12/15/2025
            reward = -1.0
            if (np.abs(new_err) < np.abs(curr_err)):
                reward = 1.0
            '''
            # linear reward structure 12/16/2025
            reward = 0
            if (max_steer < new_err < max_steer):
                reward = (1 - np.abs(new_err)/max_steer)
            else:
                reward = -1.0*(np.abs(new_err) - max_steer)/(np.pi - max_steer)
            '''
            total_reward += reward

            if (np.abs(new_err) < max_steer):
                done = True
                reward += 10
                print('angle error within max_steer range!')

            print('T: %d, S: %03d, C Dir: %0.4f, c err: %0.4f, Steer: %0.4f, N Dir: %0.4f, rwd: %0.4f, val: %0.4f' \
                % (trials,steps,dir*180/np.pi,curr_err*180/np.pi,steer*180/np.pi,new_dir*180/np.pi,reward,value_np))
            dir = new_dir

            # plot results

            #update(self, log_probs_t, values_t, curr_obs, next_obs, rewards, dones):
            closs,ploss,_ = mymodel.update(log_prob,state_in_t,next_state_t,reward,done)

            step_list.append(steps)
            dir_list.append(dir*180/np.pi)
            curr_err_list.append(np.abs(curr_err)*180/np.pi)
            new_dir_list.append(new_dir*180/np.pi)
            steer_list.append(steer*180/np.pi)
            reward_list.append(reward)
            value_list.append(value_np)
            mu_list.append(mu_np)
            std_list.append(std_np)
            closs_list.append(closs)
            ploss_list.append(ploss)

            # plot metrics
            metrics = {
                "Curr Dir": dir_list,
                "Dir err": curr_err_list,
                "Steer": steer_list,
                "Reward": reward_list,
                "value": value_list,
                "mean": mu_list,
                "stdev": std_list,
                "critic loss": closs_list,
                "actor loss": ploss_list
            }

            if (steps == 0):
                fig,axes = create_figure(trials,metrics)

            plot_title = 'Training Metrics (Trial %d, Step %d)' % (trials, steps)
            plot_trial_metrics_subplots(
                fig,
                axes,
                step_data=step_list,
                metrics=metrics,
                title=plot_title,
                normalize=False
            )

            if (done):
                break

        trial_steps_list = steps
        if (steps < trial_max_steps):
            trial_max_steps = steps
            print ('trial %d had new min steps of %d' % (trials, trial_max_steps))
        if (steps == max_steps-1):
            print ('max steps reached without an answer!')
        print('trial %d done!\n' % trials)

    print('done!')
    print('average steps: %0.2f' % np.mean(np.array(trial_steps_list)))


