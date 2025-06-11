# model.py
import torch
import torch.nn as nn

class BlackJackAgent(nn.Module):
    def __init__(self, obs_size=3, hidden_size=10, output_size=2):
        super(BlackJackAgent, self).__init__()
        self.layer_1 = nn.Linear(obs_size, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, output_size)
        self.action_probs_activation_layer = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = torch.relu(self.layer_1(x))
        logits = self.layer_2(x)
        return logits       # later use nn.Softmax to get probabilities

    def get_action_probs(self, logits):
        """Get the probabilities of each action."""
        return self.action_probs_activation_layer(logits)
    
    def sample_best_action(self, obs):
        """Get the deterministic action with the highest probability
        for a given observation.
        
        Parameters:
            obs (torch.tensor): the agent's current observable state in the playable environment. Expected shape is either `(num_features,)` for a single observation
            or `(batch_size, num_features)` for a batch of observations.
        
        Returns:
            action (int or torch.tensor): 
                - If `obs` is a single observation (i.e., `obs.dim() == 1`), returns a scalar `int` representing the chosen action. 

                - If `obs` is a batch of observations (i.e., `obs.dim() > 1`),
                returns a `torch.Tensor` of `int`s, where each element is the
                chosen action for the corresponding observation in the batch"""
        # Ensure observation is a tensor and has a batch dimension if it's a single observation
        if obs.dim() == 1:
            obs = obs.unsqueeze(0) # Add a batch dimension if it's a single observation

        logits = self.forward(obs)
        probs = self.get_action_probs(logits)
        action = torch.argmax(probs, dim=1) 
        if obs.size(0) == 1:    # This method checks if there is only 1 element in a 1-D tensor
            return action.item() # Returns a Python scalar for a single observation
        else:
            return action # Returns a tensor of actions for a batch