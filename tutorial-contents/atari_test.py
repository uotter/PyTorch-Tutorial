"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
More about Reinforcement learning: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/

Dependencies:
torch: 0.4
gym: 0.8.1
numpy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01  # learning rate
EPSILON = 0.9  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 100000
MEMORY_CAPACITY_DISCOUNT = 0.2
env = gym.make('Breakout-ram-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(),
                              int) else env.action_space.sample().shape  # to confirm the shape


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:  # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:  # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.data.numpy()


dqn = DQN()

print('\nCollecting experience...')
loss_list = []
reward_list = []
if os.path.exists("../images/"):
    pass
else:
    os.makedirs("../images/")
if os.path.exists("../models/"):
    pass
else:
    os.makedirs("../models/")
if os.path.exists("../records/"):
    pass
else:
    os.makedirs("../records/")
for i_episode in range(2000000):
    s = env.reset()
    ep_r = 0
    last_save = 0

    if i_episode % 1000 == 0 and i_episode != 0 and dqn.memory_counter > MEMORY_CAPACITY * MEMORY_CAPACITY_DISCOUNT:
        last_save = i_episode
        plt.subplot(211)
        line_loss, = plt.plot(np.array(range(len(loss_list))), loss_list, 'r-', lw=2)
        plt.legend(handles=[line_loss], labels=['loss'], loc='best')
        plt.title("loss curve")
        plt.subplot(212)
        reward_loss, = plt.plot(np.array(range(len(reward_list))), reward_list, 'r-', lw=2)
        plt.legend(handles=[reward_loss], labels=['reward'], loc='best')
        plt.title("reward curve")
        plt.savefig('../images/dqn_{}'.format(i_episode) + '.png')
        torch.save(dqn.eval_net, "../models/dqn_{}.pkl".format(i_episode))
    while True:
        # env.render()
        a = dqn.choose_action(s)

        # take action
        s_, r, done, info = env.step(a)

        # modify the reward
        # x, x_dot, theta, theta_dot = s_
        # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        # r = r1 + r2

        dqn.store_transition(s, a, r, s_)

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY * MEMORY_CAPACITY_DISCOUNT:
            step_loss = dqn.learn()
            loss_list.append(step_loss)
            if done:
                reward_list.append(ep_r)
                print('last_save: ', last_save, '| Ep: ', i_episode, '| Ep_r: ', round(ep_r, 2), '| Ep_loss: ',
                      step_loss)

        if done:
            break
        s = s_
with open("../records/loss.txt","w") as f:
    for loss in loss_list:
        f.write(str(loss) + "\n")
with open("../records/reward.txt","w") as f:
    for reward in reward_list:
        f.write(str(reward) + "\n")
