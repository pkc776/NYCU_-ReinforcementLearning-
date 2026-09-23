# Spring 2025, 535514 Reinforcement Learning
# HW2: DDPG - HalfCheetah with Clipped Double Q (CDQ)

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
import random
import os
import time
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("./tb_cdq_cheetah")

env_name = 'HalfCheetah-v4'
env = gym.make(env_name)
random_seed = 10
np.random.seed(random_seed)
torch.manual_seed(random_seed)
env.reset(seed=random_seed)

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class OUNoise:
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale

class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, action_space.shape[0])
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        x = F.relu(self.linear1(inputs))
        x = F.relu(self.linear2(x))
        action = self.tanh(self.linear3(x))
        return action

class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(num_inputs + action_space.shape[0], hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

    def forward(self, inputs, actions):
        x = torch.cat([inputs, actions], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        q_value = self.linear3(x)
        return q_value

class DDPG_CDQ(object):
    def __init__(self, num_inputs, action_space, gamma=0.99, tau=0.005, hidden_size=256, lr_a=1e-4, lr_c=1e-3):
        self.num_inputs = num_inputs
        self.action_space = action_space
        self.gamma = gamma
        self.tau = tau
        self.policy_noise_std = 0.2
        self.noise_clip = 0.5

        self.actor = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr_a)

        self.critic1 = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic2 = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic1_target = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic2_target = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic1_optim = Adam(self.critic1.parameters(), lr=lr_c)
        self.critic2_optim = Adam(self.critic2.parameters(), lr=lr_c)

        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic1_target, self.critic1)
        self.hard_update(self.critic2_target, self.critic2)

    def select_action(self, state, action_noise=None):
        self.actor.eval()
        mu = self.actor(Variable(state))
        mu = mu.data
        if action_noise is not None:
            mu += torch.Tensor(action_noise.noise())
        return mu.clamp(-1.0, 1.0)

    def update_parameters(self, batch):
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action).squeeze(1))
        reward_batch = Variable(torch.cat(batch.reward))
        mask_batch = Variable(torch.cat(batch.mask))
        next_state_batch = Variable(torch.cat(batch.next_state))

        with torch.no_grad():
            noise = torch.clamp(torch.randn_like(action_batch) * self.policy_noise_std, -self.noise_clip, self.noise_clip)
            next_action = self.actor_target(next_state_batch) + noise
            next_action = next_action.clamp(-1.0, 1.0)

            target_q1 = self.critic1_target(next_state_batch, next_action)
            target_q2 = self.critic2_target(next_state_batch, next_action)
            target_q = torch.min(target_q1, target_q2)
            expected_q = reward_batch + (mask_batch * self.gamma * target_q)

        current_q1 = self.critic1(state_batch, action_batch)
        current_q2 = self.critic2(state_batch, action_batch)
        value_loss1 = F.mse_loss(current_q1, expected_q.detach())
        value_loss2 = F.mse_loss(current_q2, expected_q.detach())

        self.critic1_optim.zero_grad()
        value_loss1.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        value_loss2.backward()
        self.critic2_optim.step()

        policy_loss = -self.critic1(state_batch, self.actor(state_batch)).mean()
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic1_target, self.critic1, self.tau)
        self.soft_update(self.critic2_target, self.critic2, self.tau)

        return (value_loss1.item() + value_loss2.item()) / 2, policy_loss.item()

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def save_model(self, env_name, suffix="", actor_path=None, critic_path_prefix=None):
        local_time = time.localtime()
        timestamp = time.strftime("%m%d%Y_%H%M%S", local_time)
        if not os.path.exists('preTrained/'):
            os.makedirs('preTrained/')

        if actor_path is None:
            actor_path = f"preTrained/ddpg_actor_{env_name}_{timestamp}{suffix}"
        if critic_path_prefix is None:
            critic1_path = f"preTrained/ddpg_critic1_{env_name}_{timestamp}{suffix}"
            critic2_path = f"preTrained/ddpg_critic2_{env_name}_{timestamp}{suffix}"
        else:
            critic1_path = f"{critic_path_prefix}_1"
            critic2_path = f"{critic_path_prefix}_2"

        print(f"Saving models to {actor_path}, {critic1_path}, and {critic2_path}")
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic1.state_dict(), critic1_path)
        torch.save(self.critic2.state_dict(), critic2_path)

    def load_model(self, actor_path, critic1_path, critic2_path):
        print(f"Loading models from {actor_path}, {critic1_path}, and {critic2_path}")
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic1.load_state_dict(torch.load(critic1_path))
        self.critic2.load_state_dict(torch.load(critic2_path))

def train():
    num_episodes = 200
    batch_size = 256
    replay_size = 1000000
    total_numsteps = 0
    max_steps = 500000
    ewma_reward = 0

    agent = DDPG_CDQ(env.observation_space.shape[0], env.action_space)
    ounoise = OUNoise(env.action_space.shape[0])
    memory = ReplayMemory(replay_size)

    for i_episode in range(num_episodes):
        ounoise.reset()
        state_np = env.reset()
        state = torch.FloatTensor(state_np).unsqueeze(0)

        episode_reward = 0
        steps = 0
        while True:
            action = agent.select_action(state, ounoise)
            next_state_np, reward, done, _ = env.step(action.numpy()[0])
            next_state = torch.FloatTensor(next_state_np).unsqueeze(0)
            reward = torch.FloatTensor([[reward]])
            mask = torch.FloatTensor([[0.0 if done else 1.0]])

            memory.push(state, action.unsqueeze(0), mask, next_state, reward)
            state = next_state
            episode_reward += reward.item()
            total_numsteps += 1
            steps += 1

            if len(memory) > batch_size:
                for _ in range(1):
                    batch = memory.sample(batch_size)
                    batch = Transition(*zip(*batch))
                    actor_loss, critic_loss = agent.update_parameters(batch)

            if done or total_numsteps >= max_steps:
                break

        ewma_reward = 0.05 * episode_reward + (1 - 0.05) * ewma_reward
        print(f"Episode: {i_episode}, Steps: {steps}, Total Steps: {total_numsteps}, Reward: {episode_reward:.2f}, EWMA: {ewma_reward:.2f}")
        writer.add_scalar("reward/episode", episode_reward, i_episode)
        writer.add_scalar("reward/ewma", ewma_reward, i_episode)

        if total_numsteps >= max_steps:
            break

    agent.save_model(env_name, '.pth')
    env.close()

if __name__ == '__main__':
    train()
