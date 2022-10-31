import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from config import Config


class Actor(nn.Module):
    def __init__(self, state_space, action_space):
        super(Actor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_space, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_space),
            nn.Tanh()
        )

    def forward(self, state):
        output = self.model(state)
        return output


class ActorCNN(nn.Module):
    def __init__(self, state_space, action_space):
        super(ActorCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=2)
        )
        self.linear = nn.Sequential(
            nn.Linear(state_space, 16),
            nn.ReLU()
        )
        self.output_layer = nn.Sequential(
            nn.Linear(1264, 256),
            nn.ReLU(),
            nn.Linear(256, action_space)
        )

    def forward(self, state):
        pos, img = state
        encoded1 = self.linear(pos)
        encoded2 = self.conv(img)
        encoded = torch.concat([encoded1.view(encoded1.shape[0], -1), encoded2.view(encoded2.shape[0], -1)], dim=1)
        output = self.output_layer(encoded)
        return output


class Critic(nn.Module):
    def __init__(self, state_space, action_space):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_space + action_space, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        # output = self.state(state)
        output = self.model(torch.cat([state, action], dim=1))
        return output


class CriticCNN(nn.Module):
    def __init__(self, state_space, action_space, n_agent=2):
        super(CriticCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=n_agent, out_channels=8, kernel_size=(1, 3, 3), stride=2),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=2),
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(1, 5, 5), stride=2),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 5, 5), stride=2)
        )
        self.pos_encoder = nn.Sequential(
            nn.Linear(state_space, 16),
            nn.ReLU()
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_space, 16),
            nn.ReLU()
        )
        self.output_layer = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        pos, img = state
        encoded1 = self.conv(img)
        encoded2 = self.pos_encoder(pos.view(pos.shape[0], -1))
        encoded3 = self.action_encoder(action)
        encoded = torch.concat([encoded1.view(img.shape[0], -1), encoded2, encoded3], dim=1)
        # print(encoded1.shape, encoded2.shape, encoded3.shape, encoded.shape)
        output = self.output_layer(encoded)
        return output


class MADDPG:
    def __init__(self, state_size, action_size, n_agent, gamma=0.99,
                 lr_actor=0.001, lr_critic=0.005, update_freq=300):
        self.state_size = state_size
        self.action_size = action_size
        self.n_agent = n_agent
        self.gamma = gamma
        self.update_freq = update_freq
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using device {}".format(self.device))

        self.actors = [ActorCNN(state_size, action_size).to(self.device) for _ in range(n_agent)]
        self.actors_target = [ActorCNN(state_size, action_size).to(self.device) for _ in range(n_agent)]

        self.critics = [CriticCNN(state_size * n_agent, action_size * n_agent, n_agent).to(self.device) for _ in range(n_agent)]
        self.critics_target = [CriticCNN(state_size * n_agent, action_size * n_agent, n_agent).to(self.device) for _ in range(n_agent)]
        # self.critic = Critic(state_size * n_agent, action_size * n_agent).to(self.device)
        # self.critic_target = Critic(state_size * n_agent, action_size * n_agent).to(self.device)

        [actor_target.eval() for actor_target in self.actors_target]
        [critic_target.eval() for critic_target in self.critics_target]

        self.actors_optim = [optim.Adam(actor.parameters(), lr_actor) for actor in self.actors]
        self.critics_optim = [optim.Adam(critic.parameters(), lr_critic) for critic in self.critics]
        # self.actor_optim = optim.Adam(sum([list(actor.parameters()) for actor in self.actors]))
        # self.critic_optim = optim.Adam(sum([list(critic.parameters()) for critic in self.critics]))
        self.update_target()

        self.last_all_state = None
        self.last_all_state_next = None
        self.last_all_action = None

        self.steps = 0

    def update_target(self):
        for i in range(self.n_agent):
            self.critics_target[i].load_state_dict(self.critics[i].state_dict())
            self.actors_target[i].load_state_dict(self.actors[i].state_dict())

    def to_tensor(self, inputs):
        if torch.is_tensor(inputs):
            return inputs
        return torch.FloatTensor(inputs).to(self.device)

    def choose_action(self, states):
        actions = [actor([self.to_tensor(state[0]), self.to_tensor(state[1]).unsqueeze(0)]).squeeze().detach().cpu().numpy()
                   for actor, state in zip(self.actors, states)]
        return actions

    def learn(self, s, a, r, sn, d):
        # print("update")
        states_locs = [self.to_tensor(state[0]) for state in s]
        states_imgs = [self.to_tensor(state[1]) for state in s]
        # states_locs = self.to_tensor(s[0])
        # states_imgs = self.to_tensor(s[1])
        actions = [self.to_tensor(action) for action in a]
        rewards = [self.to_tensor(reward) for reward in r]
        rewards = [(reward - torch.mean(reward)) / (torch.std(reward) + 1e-4) for reward in rewards]
        states_next_locs = [self.to_tensor(state_next[0]) for state_next in sn]
        states_next_imgs = [self.to_tensor(state_next[1]) for state_next in sn]
        # print(s[0][0].shape, s[0][1][0].shape, states_locs[0].shape, states_imgs[0].shape)
        dones = [self.to_tensor(done.astype(int)) for done in d]
        comm = np.random.rand() > Config.comm_fail_prob
        if self.last_all_action is None or self.last_all_state is None or self.last_all_state_next is None or comm:
            all_state_imgs = torch.stack(states_imgs)
            all_state_locs = torch.cat(states_locs, dim=1)
            all_action = torch.cat(actions, dim=1)
            all_state_next_imgs = torch.stack(states_next_imgs)
            all_state_next_locs = torch.cat(states_next_locs, dim=1)
        else:
            all_state_locs, all_state_imgs = self.last_all_state
            all_action = self.last_all_action
            all_state_next_locs, all_state_next_imgs = self.last_all_state_next
        all_state_imgs = torch.swapaxes(all_state_imgs, 0, 1)
        all_state_next_imgs = torch.swapaxes(all_state_next_imgs, 0, 1)
        # print(all_state_imgs.shape, all_state_locs.shape)
        actor_losses = 0
        for i in range(self.n_agent):
            cur_action = all_action.clone()
            action = self.actors[i]([states_locs[i], states_imgs[i]])
            action_size = action.shape[1]
            cur_action[:, action_size * i: action_size * (i + 1)] = action
            actor_loss = -torch.mean(self.critics[i]([all_state_locs, all_state_imgs], cur_action))
            actor_losses += actor_loss

        actions_next = [actor_target([state_next_loc, state_next_img]).detach() for
                        state_next_loc, state_next_img, actor_target in
                        zip(states_next_locs, states_next_imgs, self.actors_target)]
        all_action_next = torch.cat(actions_next, dim=1)
        critic_losses = 0
        for i in range(self.n_agent):
            next_value = self.critics_target[i]([all_state_next_locs, all_state_next_imgs], all_action_next)
            Q = self.critics[i]([all_state_locs, all_state_imgs], all_action)
            target = rewards[i] + self.gamma * next_value.detach()
            critic_loss = F.mse_loss(Q, target)
            critic_losses += critic_loss

        # actor
        # self.actor_optim.zero_grad()
        [actor_optim.zero_grad() for actor_optim in self.actors_optim]
        actor_losses.backward()
        [nn.utils.clip_grad_norm_(actor.parameters(), 0.3) for actor in self.actors]
        # self.actor_optim.step()
        [actor_optim.step() for actor_optim in self.actors_optim]
        # critic
        # self.critic_optim.zero_grad()
        [critic_optim.zero_grad() for critic_optim in self.critics_optim]
        critic_losses.backward()
        [nn.utils.clip_grad_norm_(critic.parameters(), 0.3) for critic in self.critics]
        # self.critic_optim.step()
        [critic_optim.step() for critic_optim in self.critics_optim]

        # update target networks
        if self.steps % self.update_freq == 0:
            self.update_target()
        self.steps += 1
        # print(actor_losses, critic_losses)

        return (actor_losses + critic_losses).item()

    def save_model(self, path):
        save_content = {}
        for i in range(self.n_agent):
            save_content['actor_{}'.format(i)] = self.actors[i].state_dict()
            save_content['critic_{}'.format(i)] = self.critics[i].state_dict()
        torch.save(save_content, path)

    def load_model(self, path):
        saved_content = torch.load(path)
        for i in range(self.n_agent):
            self.actors[i].load_state_dict(saved_content['actor_{}'.format(i)])
            self.critics[i].load_state_dict(saved_content['critic_{}'.format(i)])
        # self.update_target()