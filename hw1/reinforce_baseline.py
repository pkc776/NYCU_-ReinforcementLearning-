import os
import gym
from itertools import count
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

# Define named tuple for storing actions
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# TensorBoard writer
writer = SummaryWriter("./tb_lunar_baseline")

class Policy(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(Policy, self).__init__()
        self.shared_fc = nn.Linear(observation_dim, 128)
        self.action_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        if state.dtype != torch.float32:
            state = state.float()
        x = F.relu(self.shared_fc(state))
        action_probs = F.softmax(self.action_head(x), dim=-1)
        state_value = self.value_head(x)
        return action_probs, state_value

    def select_action(self, state):
        state = torch.from_numpy(state)
        probs, state_value = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        return action.item()

    def calculate_loss(self, gamma=0.99):
        R = 0
        returns = []
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        policy_losses = []
        value_losses = []
        for (log_prob, value), R in zip(self.saved_actions, returns):
            advantage = R - value.item()  # baseline: state value
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([[R]])))

        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        return loss

    def clear_memory(self):
        del self.rewards[:]
        del self.saved_actions[:]

def train(env_name='LunarLander-v2', lr=1e-3):
    env = gym.make(env_name)
    model = Policy(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ewma_reward = 0

    for i_episode in range(1, 1001):
        state = env.reset()
        ep_reward = 0

        for t in range(1000):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        loss = model.calculate_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.clear_memory()

        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward

        print(f"Episode {i_episode}\tReward: {ep_reward:.2f}\tEWMA Reward: {ewma_reward:.2f}")
        writer.add_scalar('Reward/episode', ep_reward, i_episode)
        writer.add_scalar('EWMA Reward/episode', ewma_reward, i_episode)
        writer.add_scalar('Loss/episode', loss.item(), i_episode)

        if ewma_reward > env.spec.reward_threshold:
            if not os.path.exists('./preTrained'):
                os.mkdir('./preTrained')
            torch.save(model.state_dict(), f"./preTrained/LunarLander_{lr}.pth")
            print("Solved LunarLander early! Saving model.")
            return  # ← 提早結束訓練

    # 🔽 這裡是補充的收尾儲存邏輯
    if not os.path.exists('./preTrained'):
        os.mkdir('./preTrained')
    torch.save(model.state_dict(), f"./preTrained/LunarLander_{lr}_final.pth")
    print("Training complete (2000 episodes). Final model saved.")


def test(env_name='LunarLander-v2', model_path='LunarLander_0.001_final.pth'):
    env = gym.make(env_name)
    model = Policy(env.observation_space.shape[0], env.action_space.n)
    model.load_state_dict(torch.load(f'./preTrained/{model_path}'))
    model.eval()

    for ep in range(10):
        state = env.reset()
        total_reward = 0
        for t in range(1000):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            env.render()
            if done:
                break
        print(f"Test Episode {ep + 1}: Total Reward = {total_reward:.2f}")
    env.close()

if __name__ == '__main__':
    torch.manual_seed(42)
    train()
    test(model_path='LunarLander_0.001_final.pth')

