# main.py
import torch
import streamlit as st
from scripts import BlackJackAgent
import os

agent = BlackJackAgent()    # Create agent instance
# Dynamically create the path to the model's weights 
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Get directory of current running file
weights_file = os.path.join(BASE_DIR, "model_weights", "blackjack_policy_model.pth") # create the full path to the model weights
agent.load_state_dict(torch.load(weights_file)) # Load the agent's model weights


# User inputs
player_sum = st.slider("Player's Hand Sum", min_value=4, max_value=21, value=15)
dealer_sum = st.slider("Dealer's Visible Card", min_value=1, max_value=11, value=10)
usable_ace = st.selectbox("Usable Ace?", options=[0, 1], format_func=lambda x: "Yes" if x else "No")

# Prepare input tensor
raw_inputs = (player_sum, dealer_sum, usable_ace)
inputs = torch.tensor(raw_inputs, dtype=torch.float32).unsqueeze(0)

# Predict action
if st.button("Get Agent's Action"):
    action = agent.sample_best_action(inputs)
    action_str = "Hit" if action == 1 else "Stick"
    st.success(f"Agent recommends: **{action_str}**")