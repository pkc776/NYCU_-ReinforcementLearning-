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
import torch.optim.lr_scheduler as Scheduler
from torch.utils.tensorboard import SummaryWriter

# Define a useful tuple
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# Define a tensorboard writer
writer = SummaryWriter("./tb_record_1")

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 128

        # Network layers
        self.shared_fc = nn.Linear(self.observation_dim, self.hidden_size)
        self.action_head = nn.Linear(self.hidden_size, self.action_dim)
        self.value_head = nn.Linear(self.hidden_size, 1)

        # Initialize weights
        for layer in [self.shared_fc, self.action_head, self.value_head]:
            nn.init.kaiming_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        if state.dtype != torch.float32:
            state = state.float()
        x = F.relu(self.shared_fc(state))
        action_prob = F.softmax(self.action_head(x), dim=-1)
        state_value = self.value_head(x)
        return action_prob, state_value

    def select_action(self, state):
        state = torch.from_numpy(state)
        probs, state_value = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        return action.item()

    def calculate_loss(self, gamma=0.999):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = []
        value_losses = []
        returns = []

        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([[R]])))

        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        return loss

    def clear_memory(self):
        del self.rewards[:]
        del self.saved_actions[:]

def train(lr=0.01):
    model = Policy()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ewma_reward = 0

    for i_episode in count(1):
        state = env.reset()
        ep_reward = 0
        t = 0

        for t in range(1, 10000):
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
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(i_episode, t, ep_reward, ewma_reward))

        writer.add_scalar('Reward/episode', ep_reward, i_episode)
        writer.add_scalar('EWMA Reward/episode', ewma_reward, i_episode)
        writer.add_scalar('Loss/episode', loss.item(), i_episode)
        writer.add_scalar('Learning Rate', lr, i_episode)

        if ewma_reward > env.spec.reward_threshold:
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), './preTrained/CartPole_{}.pth'.format(lr))
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, t))
            break

def test(name, n_episodes=10):
    model = Policy()
    model.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    render = True
    max_episode_len = 10000

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        running_reward = 0
        for t in range(max_episode_len + 1):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                env.render()
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()

if __name__ == '__main__':
    random_seed = 10
    lr = 0.01
    env = gym.make('CartPole-v0')
    env.seed(random_seed)
    torch.manual_seed(random_seed)
    train(lr)
    test(f'CartPole_{lr}.pth')
