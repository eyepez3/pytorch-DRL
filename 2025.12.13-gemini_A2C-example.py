import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

# --- 1. Define the Actor Network Class ---
class ContinuousActor(nn.Module):
    """
    Actor network for continuous action spaces, implementing a Gaussian Policy.
    It outputs the mean (mu) and the log standard deviation (log_std) of the action distribution.
    """
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(ContinuousActor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Output layer for the Mean (mu) of the action distribution
        self.mu_head = nn.Linear(hidden_size, action_dim)
        
        # Output layer for the Log Standard Deviation (log_std)
        # Note: We output log_std for stability, not std directly.
        self.log_std_head = nn.Linear(hidden_size, action_dim)

        # Initialize small weights for the final layers to start with a modest exploration range
        nn.init.uniform_(self.log_std_head.weight, -1e-3, 1e-3)

        # Set the limits for log_std (hyperparameters for stability)
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # --- Mean (mu) Output ---
        # Activation: tanh (or a scaled tanh) for bounded action spaces [-1, 1]
        mu = torch.tanh(self.mu_head(x))
        
        # --- Log Standard Deviation (log_std) Output ---
        # Activation: Linear (Identity) for log_std
        log_std = self.log_std_head(x)
        
        # Clip log_std to a safe range to maintain numerical stability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mu, log_std

    def sample(self, state):
        """
        Performs the forward pass and then samples an action from the resulting distribution.
        This is what the actor uses during exploration/training.
        """
        # 1. Get the parameters (mu, log_std)
        mu, log_std = self.forward(state)
        
        # 2. Transform log_std to standard deviation (std)
        std = torch.exp(log_std)
        
        # 3. Create the Normal distribution
        normal = Normal(mu, std)
        
        # 4. Sample a 'raw' action (action_unbounded) from the distribution
        action_unbounded = normal.rsample()  # rsample is used for reparameterization trick (important for SAC)
        
        # 5. Apply tanh to squash the action to the desired bounds [-1, 1]
        action = torch.tanh(action_unbounded)
        
        # 6. Calculate the log probability (used for policy gradient calculation)
        # Note: The log_prob must be corrected due to the tanh squashing transformation
        log_prob = normal.log_prob(action_unbounded)
        corr = torch.log(1 - action.pow(2) + 1e-6) # Transformation correction term
        log_prob -= corr 
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, mu, log_std

    def select_action(self, state):
        """
        Selects the deterministic action (mean of the distribution) for testing/evaluation.
        """
        mu, _ = self.forward(state)
        # The mean is already tanh-squashed in the forward pass
        return mu.detach().cpu().numpy().flatten()


# --- 2. Example Usage ---
if __name__ == '__main__':
    STATE_DIM = 2   # Example state: cart-pole position, velocity, etc.
    ACTION_DIM = 1  # Example continuous action: motor 1 torque, motor 2 torque
    
    # 1. Initialize the Actor
    actor = ContinuousActor(STATE_DIM, ACTION_DIM)
    
    print(f"Actor Network Structure:\n{actor}")

    # 2. Create a dummy state input (batch size of 1)
    # Typically, you process batches of states from the environment or replay buffer
    dummy_state = torch.randn(1, STATE_DIM) 
    
    # --- A. Sampling for Training/Exploration ---
    print("\n--- A. Sampling for Training ---")
    
    # action: the squashed action in [-1, 1]
    # log_prob: the log likelihood of taking that action
    action, log_prob, mu, log_std = actor.sample(dummy_state)
    
    print(f"Input State Shape: {dummy_state.shape}")
    print(f"Output Action (Sampled): {action.detach().numpy()}")
    print(f"Action Shape: {action.shape}")
    print(f"Mean (mu) Output: {mu.detach().numpy()}")
    print(f"Log Std Dev (log_std) Output: {log_std.detach().numpy()}")
    print(f"Log Probability: {log_prob.detach().item():.4f}")
    
    # Verification: Check that the sampled action is within the [-1, 1] bounds
    is_bounded = torch.all((action >= -1.0) & (action <= 1.0))
    print(f"Is action bounded [-1, 1]? {is_bounded.item()}")

    # --- B. Selecting Deterministic Action for Evaluation ---
    print("\n--- B. Selecting Deterministic Action for Evaluation ---")
    
    # The deterministic action is simply the mean (mu)
    deterministic_action = actor.select_action(dummy_state)
    
    print(f"Deterministic Action (Mu): {deterministic_action}")