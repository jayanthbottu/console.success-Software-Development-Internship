"""
Deep Q-Network Agent for Traffic Control
Implements DQN with Experience Replay and Target Networks
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from config import *

class DQNetwork(nn.Module):
    """Deep Q-Network architecture"""
    
    def __init__(self):
        super(DQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(STATE_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(HIDDEN_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_SIZE)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    """DQN Agent with experience replay and target network"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Networks
        self.policy_net = DQNetwork().to(self.device)
        self.target_net = DQNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.SmoothL1Loss()  # Huber loss
        
        # Replay memory
        self.memory = deque(maxlen=MEMORY_SIZE)
        
        # Exploration
        self.epsilon = EPSILON_START
        self.steps_done = 0
    
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, ACTION_SIZE - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self):
        """
        Perform one step of training using experience replay
        """
        if len(self.memory) < BATCH_SIZE:
            return None
        
        # Sample batch
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions).squeeze()
        
        # Target Q values using target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
        self.steps_done += 1
        
        # Update target network
        if self.steps_done % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def save_model(self, path):
        """Save model weights"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', EPSILON_END)
        self.steps_done = checkpoint.get('steps_done', 0)
        print(f"Model loaded from {path}")