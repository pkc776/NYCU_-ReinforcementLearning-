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

# Named tuple for storing actions
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# TensorBoard writer
writer = SummaryWriter("./tb_lunar_gae")

class Policy(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(Policy, self).__init__()
        self.shared_fc = nn.Linear(observation_dim, 128)
        self.action_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []
        self.state_values = []

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
        self.state_values.append(state_value.item())
        return action.item()

    def clear_memory(self):
        del self.rewards[:]
        del self.saved_actions[:]
        del self.state_values[:]

class GAE:
    def __init__(self, gamma=0.99, lambda_=0.95):
        self.gamma = gamma
        self.lambda_ = lambda_

    def compute_advantages(self, rewards, values, done):
        advantages = []
        gae = 0
        values = values + [0]  # Add terminal value
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - done) - values[t]
            gae = delta + self.gamma * self.lambda_ * (1 - done) * gae
            advantages.insert(0, gae)
        return advantages

def calculate_loss(model, gamma, gae_lambda):
    R = 0
    returns = []
    for r in model.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-5)

    # Compute advantages using GAE
    gae = GAE(gamma=gamma, lambda_=gae_lambda)
    advantages = gae.compute_advantages(model.rewards, model.state_values, done=0)
    advantages = torch.tensor(advantages)

    policy_losses = []
    value_losses = []
    for (log_prob, value), adv, R in zip(model.saved_actions, advantages, returns):
        policy_losses.append(-log_prob * adv)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([[R]])))

    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    return loss
def train(lr=1e-3, gae_lambda=0.95):
    env = gym.make('LunarLander-v2')
    model = Policy(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ewma_reward = 0

    for i_episode in range(1, 501):
        state = env.reset()
        ep_reward = 0

        for t in range(1000):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        loss = calculate_loss(model, gamma=0.99, gae_lambda=gae_lambda)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.clear_memory()

        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward

        print(f"[Lambda={gae_lambda}] Episode {i_episode}\tReward: {ep_reward:.2f}\tEWMA: {ewma_reward:.2f}")
        writer.add_scalar(f'Lambda_{gae_lambda}/Reward', ep_reward, i_episode)
        writer.add_scalar(f'Lambda_{gae_lambda}/EWMA', ewma_reward, i_episode)
        writer.add_scalar(f'Lambda_{gae_lambda}/Loss', loss.item(), i_episode)

        if ewma_reward > env.spec.reward_threshold:
            if not os.path.exists('./preTrained'):
                os.mkdir('./preTrained')
            torch.save(model.state_dict(), f"./preTrained/LunarLander_GAE_lambda{gae_lambda}_early.pth")
            print(f"Solved with GAE (lambda={gae_lambda})! Early save.")
            break

    # Always save the final model
    if not os.path.exists('./preTrained'):
        os.mkdir('./preTrained')
    torch.save(model.state_dict(), f"./preTrained/LunarLander_GAE_lambda{gae_lambda}_final.pth")
    print(f"Training complete for lambda={gae_lambda}. Final model saved.")


def test(model_path, env_name='LunarLander-v2'):
    env = gym.make(env_name)
    model = Policy(env.observation_space.shape[0], env.action_space.n)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for ep in range(5):
        state = env.reset()
        total_reward = 0
        for _ in range(1000):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            env.render()
            if done:
                break
        print(f"Test Episode {ep + 1}: Reward = {total_reward:.2f}")
    env.close()

if __name__ == '__main__':
    torch.manual_seed(42)
    for lambda_ in [0.90, 0.95, 0.99]:
        train(lr=1e-3, gae_lambda=lambda_)
    # test("./preTrained/LunarLander_GAE_lambda0.95.pth")