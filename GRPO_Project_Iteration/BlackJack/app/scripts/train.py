# %% [markdown]
# **BLACKJACK AGENT**
# **Part 3**
# 
# * Adding vectorized BlackJack Environment
# * Adding Device Agnostic code (GPU Training)
# * Attempting to use Softmax (Categorical Distribution) implementation instead of Sigmoid (Binary Bernoulli Distribution)

# %% [markdown]
# **Results**
# 
# * Still very slow during training

# %% [markdown]
# # Imports

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import gymnasium as gym
import argparse
import time

# %% [markdown]
# # Agent

# %%
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

# %% [markdown]
# # Training Loop

# %%
def training_blackjack_agent(epochs=50, learning_rate=0.0001, batch_size=64, gamma=0.99, k_epochs=64, epsilon=0.2, beta_kl=0.01, max_grad_norm=0.5, entropy_coeff=0.01, log_iterations=10, device="cpu", num_envs=16) -> BlackJackAgent: 
    print(f"Training BlackJack Agent's Policy on {device} with {epochs} epochs, {learning_rate} learning rate, batch size {batch_size}, and KL beta {beta_kl}.")

    vec_env = gym.make_vec("Blackjack-v1", num_envs=num_envs, sab=True) # `sab=True` uses the Sutton & Barto version

    # steps_per_env_per_rollout = batch_size // num_envs if batch_size % num_envs == 0 else (batch_size // num_envs) + 1

    New_Policy = BlackJackAgent().to(device)   # STEP 3 || 
    optimizer = optim.Adam(params=New_Policy.parameters(), lr=learning_rate)


    for epoch in tqdm(range(epochs), desc=f"Main Epoch (Outer Loop)", leave=False):     # STEP 4 || 
        # STEP 5 || Sample a batch D_b from D --> OMITTED 
        # STEP 6 || Update the old policy model PI old <- PI new
        Policy_Old = BlackJackAgent().to(device)
        Policy_Old.load_state_dict(New_Policy.state_dict())
        Policy_Old.eval()   # Prevent Gradient tracking

        # This will store trajectories for all episodes collected in the current batch
        completed_batch_trajectories = []

        # Reset all vectorized environments
        raw_observations, infos = vec_env.reset() # observations is a numpy array of shape (num_envs, obs_dim(3))
        observations = np.stack(raw_observations, axis=1)
        dones = np.array([False] * num_envs) # Track the done status for each parallel environment
        truncateds = np.array([False] * num_envs) # Track truncated status for each parallel environment

        # Initialize current trajectories for all parallel environments
        # Each element in this list will be a dict for an *in-progress* episode in a specific env
        current_episode_trajectories = [{"states": [], "actions": [], "rewards": [], "log_probs": []} for _ in range(num_envs)]

        # --- STEP 7 Collect a Batch of Experiences Using the Old Policy---
        # Loop Agent prediction, recording trajectories to lists:
        episodes_collected_in_batch = 0
        max_steps_per_batch_limit = batch_size * 5 # A safety limit to prevent infinite loops if episodes are very long
        current_total_steps = 0

        while episodes_collected_in_batch < batch_size and current_total_steps < max_steps_per_batch_limit:
            obs_tensor = torch.tensor(observations, dtype=torch.float32).to(device)

            with torch.no_grad():
                logits = Policy_Old(obs_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample() # Tensor of shape [1]
                log_probs = dist.log_prob(actions)
                    
            raw_next_obs, rewards, dones, truncateds, infos = vec_env.step(actions.cpu().numpy()) # actions must be on CPU for env.step()
            next_obs = np.stack(raw_next_obs, axis=1)
            current_total_steps += num_envs

            # Process data for each parallel environment
            for env_idx in range(num_envs):
                
                obs_to_append = observations[env_idx]
                if isinstance(obs_to_append, torch.Tensor):
                    obs_to_append = obs_to_append.cpu().numpy()
                # Store current_episode_trajectories
                current_episode_trajectories[env_idx]["states"].append(obs_to_append)
                current_episode_trajectories[env_idx]["actions"].append(actions[env_idx].item())
                current_episode_trajectories[env_idx]["rewards"].append(rewards[env_idx])
                current_episode_trajectories[env_idx]["log_probs"].append(log_probs[env_idx].cpu())
                
                if dones[env_idx] or truncateds[env_idx]:
                    completed_batch_trajectories.append(current_episode_trajectories[env_idx])
                    episodes_collected_in_batch += 1


                    current_episode_trajectories[env_idx] = {"states": [], "actions": [], "rewards": [], "log_probs": []}

            observations = next_obs  # Update the observation

        for env_idx in range(num_envs):
            if len(current_episode_trajectories[env_idx]["states"]) > 0:
                # If there's partial data, it means the episode was still running
                # when `batch_size` was met. You'll need to decide how to handle this.
                # For simplicity for now, we'll append them. In full PPO, you'd add
                # the value of the last state to its rewards.
                completed_batch_trajectories.append(current_episode_trajectories[env_idx])
                # Note: These might not be "full" episodes in the sense of reaching a done state,
                # but they contribute steps to your batch.

        # These lists will hold data from ALL episodes in the current batch for Advantage Calculation
        all_states = []
        all_actions = []
        all_old_log_probs = []
        all_discounted_rewards = []

        # STEP 8 || Calculate Discounted Rewards for completed trajectories
        for episode_trajectory in completed_batch_trajectories: 
            rewards = episode_trajectory["rewards"]
            states = episode_trajectory["states"]
            actions = episode_trajectory["actions"]
            log_probs = episode_trajectory["log_probs"]
            
            if not rewards:
                continue

            discounted_reward = 0
            returns_for_episode = []
            for reward in reversed(rewards):
                discounted_reward = reward + gamma * discounted_reward
                returns_for_episode.insert(0, discounted_reward)

            discounted_rewards = torch.tensor(returns_for_episode, dtype=torch.float32)
            # print(f"discounted_rewards size: {discounted_rewards.size()}")
            # Add each trajectory information for the batch
            if states:
                all_states.extend(states)
                all_actions.extend(actions)
                all_old_log_probs.extend(log_probs)
                all_discounted_rewards.extend(discounted_rewards.tolist())

        # --- IMPORTANT: Pre-tensorization checks and conversions ---
        if not all_states or not all_actions or not all_old_log_probs or not all_discounted_rewards:
            print(f"Warning: Epoch {epoch + 1}: Insufficient data collected for optimization. "
                  f"Skipping policy update for this epoch.")
            print(f"  Counts: States={len(all_states)}, Actions={len(all_actions)}, "
                  f"LogProbs={len(all_old_log_probs)}, Rewards={len(all_discounted_rewards)}")
            continue
        # Convert all collected batch data into PyTorch tensors
        all_states_tensor = torch.tensor(np.array(all_states), dtype=torch.float32).to(device)
        all_actions_tensor = torch.tensor(all_actions, dtype=torch.long).to(device)
        # Stack individual log_prob tensors and then flatten if necessary
        all_old_log_probs_tensor = torch.tensor(all_old_log_probs, dtype=torch.float32).to(device) # Ensure it's a 1D tensor
        all_discounted_rewards_tensor = torch.tensor(all_discounted_rewards, dtype=torch.float32).to(device)

        # STEP 9 || Calculate the Advantage of each Time Step for each Trajectory using normalization
        all_advantages_tensor = (all_discounted_rewards_tensor - all_discounted_rewards_tensor.mean()) / (all_discounted_rewards_tensor.std() + 1e-8)

        # Detach these tensors from any computation graph history
        # as they represent fixed data for the policy updates in k_epochs.
        # This prevents the "RuntimeError: Trying to backward through the graph a second time".
        all_states_tensor = all_states_tensor.detach()
        all_actions_tensor = all_actions_tensor.detach()
        all_old_log_probs_tensor = all_old_log_probs_tensor.detach()
        all_advantages_tensor = all_advantages_tensor.detach()

        New_Policy.train()  # Prepare NN for updates

        # --- STEP 10 || GRPO Optimization ---
        for k_epoch in tqdm(range(k_epochs), desc=f"Epoch {epoch+1}/{epochs} (Inner K-Epochs)", leave=True):
            new_logits = New_Policy(all_states_tensor)
            new_dist = torch.distributions.Categorical(logits=new_logits)
            new_log_probs = new_dist.log_prob(all_actions_tensor)
            entropy = new_dist.entropy().mean() # Calculate entropy for regularization

            R1_ratio = torch.exp(new_log_probs - all_old_log_probs_tensor)

            unclipped_surrogate = R1_ratio * all_advantages_tensor
            clipped_surrogate = torch.clamp(input=R1_ratio, min=1.0-epsilon, max=1.0+epsilon) * all_advantages_tensor

            policy_loss = -torch.min(unclipped_surrogate, clipped_surrogate).mean()

            # --- KL Divergence Calculation ---
            # Create distributions for old policies using the trajectory states
            with torch.no_grad():
                old_logits = Policy_Old(all_states_tensor)
            old_dist = torch.distributions.Categorical(logits=old_logits)

            # Calculate KL divergence per sample, then take the mean over the batch
            kl_div_per_sample = torch.distributions.kl.kl_divergence(p=new_dist, q=old_dist)
            kl_loss = kl_div_per_sample.mean() # Mean over the batch

            # Total Loss for GRPO
            total_loss = policy_loss + beta_kl * kl_loss - entropy_coeff * entropy

            # STEP 11 || Policy Updates
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(New_Policy.parameters(), max_grad_norm)
            optimizer.step()    # Update policy parameters using gradient ascent
        
        
        # --- 4. Logging and Evaluation ---
        if (epoch + 1) % log_iterations == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss.item():.4f}, Ratio: {R1_ratio.mean().item():.5f}, Entropy Term: {entropy:.5f}")
            # You can add more evaluation metrics here, e.g., average reward per episode
            # For Blackjack, the reward is often -1, 0, or 1.
            avg_reward = sum(sum(ep["rewards"]) for ep in completed_batch_trajectories) / len(completed_batch_trajectories) if len(completed_batch_trajectories) > 0 else 0
            print(f"Average reward per episode in batch: {avg_reward:.2f}")

    New_Policy.eval()   # Change to eval mode for evaluation


    vec_env.close() # Close the environment after training
    print("Training complete.")
    return New_Policy # Return the trained policy
# %%
def main(args):
    print("Beginning Training Script")
    
    if args.device:     # Check if the user specified to use a CPU or GPU for training
        device = args.device
    else:
        if args.use_cuda:   # Check if the user wanted to use CUDA if available.
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time=time.time()
    trained_policy = training_blackjack_agent(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size, # Significantly larger batch size recommended for stability
        k_epochs=args.k_epochs,
        epsilon=args.epsilon,
        beta_kl=args.beta_kl,
        entropy_coeff=args.entropy_coeff,
        log_iterations=args.log_iterations,
        gamma=args.gamma,
        device=device,
        num_envs=args.num_envs
    )
    end_time=time.time()

    elapsed_time= end_time - start_time
    hrs = int(elapsed_time / 3600)
    min = int((elapsed_time % 3600) / 60)
    seconds_remaining = elapsed_time - (hrs * 3600 ) - (min * 60)
    print(f"\nTraining Took: {hrs} Hours, {min} Minutes, and {seconds_remaining} Seconds")


    print("\nTesting the trained policy:")
    test_env = gym.make("Blackjack-v1", sab=True)
    total_test_rewards = 0
    num_test_episodes = 1000

    for _ in range(num_test_episodes):
        obs, _ = test_env.reset()
        done = False
        truncated = False
        episode_reward = 0
        while not done and not truncated:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action = trained_policy.sample_best_action(obs_tensor)
            obs, reward, done, truncated, _ = test_env.step(action)
            episode_reward += reward
        total_test_rewards += episode_reward

    print(f"\nAverage reward over {num_test_episodes} test episodes: {total_test_rewards / num_test_episodes:.4f}")
    test_env.close()    # safely close the gym environment after the testing and validation of the trained model

    SAVE_LOCATION = "./app/model_weights/blackjack_policy_model.pth"   # Define the model path and name of the trained model weights

    if args.save_model:     # Check if the user wants to save the trained model weights
        if args.model_output_path:     # Check if the user specified a target save location
            SAVE_LOCATION=args.model_output_path
        
        torch.save(trained_policy.parameters(), f=SAVE_LOCATION)
        print(f"Model weights saved in: {SAVE_LOCATION}")

    print("Exiting Script")

# %%
# Example usage (assuming you have a way to call this function, e.g., in a main block)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and test a BlackJack PPO agent.")

    # Add arguments
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.0003,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for training.')
    parser.add_argument('--k_epochs', type=int, default=128,
                        help='Number of policy update epochs per trajectory collection.')
    parser.add_argument('--epsilon', type=float, default=0.2,
                        help='Clipping parameter for PPO.')
    parser.add_argument('--beta_kl', type=float, default=0.01,
                        help='KL divergence coefficient (for PPO-like algorithms).')
    parser.add_argument('--entropy_coeff', type=float, default=0.001,
                        help='Entropy regularization coefficient.')
    parser.add_argument('--log_iterations', type=int, default=100,
                        help='Log training progress every N iterations.')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor for rewards.')
    parser.add_argument('--num_envs', type=int, default=16,
                        help='Number of parallel environments for training.')
    parser.add_argument('--use_cuda', action='store_true',
                        help='Use CUDA if available using this flag.')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Explicitly set device (e.g., "cpu, cuda:0", "cpu"). Overrides --use_cuda if specified.')
    parser.add_argument('--save_model', action='store_true',
                        help='Save the trained model weights using this flag.')
    parser.add_argument('--model_output_path', type=str, default='blackjack_policy_model.pth',
                        help='Path to save the trained model weights.')

    # Parse the arguments
    args = parser.parse_args()

    
    main(args)