import copy
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from agent.model import MLP, Critic
from agent.diffusion import Diffusion
from agent.vae import VAE
from agent.helpers import EMA
from agent.q_transform import *
import os
import time


class QVPO(object):
    def __init__(self,
                 args,
                 state_dim,
                 action_space,
                 memory,
                 diffusion_memory,
                 device,
                 ):
        action_dim = np.prod(action_space.shape)

        self.policy_type = args.policy_type
        if self.policy_type == 'Diffusion':
            self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, noise_ratio=args.noise_ratio,
                                   beta_schedule=args.beta_schedule, n_timesteps=args.n_timesteps, behavior_sample=args.behavior_sample,
                                   eval_sample=args.eval_sample, deterministic=args.deterministic).to(device)
            self.running_q_std = 1.0
            self.running_q_mean = 0.0
            self.beta = args.beta
            self.alpha_mean = args.alpha_mean
            self.alpha_std = args.alpha_std
            self.chosen = args.chosen
            self.q_neg = args.q_neg

            self.weighted = args.weighted
            self.aug = args.aug
            self.train_sample = args.train_sample

            self.q_transform = args.q_transform
            self.gradient = args.gradient
            self.policy_freq = args.policy_freq

            self.cut = args.cut
            self.epsilon = args.epsilon

            self.entropy_alpha = args.entropy_alpha

        else:
            raise NotImplementedError

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.diffusion_lr, eps=1e-5)

        self.memory = memory
        if not self.aug:
            self.diffusion_memory = diffusion_memory
        self.action_gradient_steps = args.action_gradient_steps

        self.action_grad_norm = action_dim * args.ratio
        self.ac_grad_norm = args.ac_grad_norm

        self.step = 0
        self.tau = args.tau
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.behavior_sample = args.target_sample
        self.update_actor_target_every = args.update_actor_target_every

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, eps=1e-5)

        self.action_dim = action_dim

        self.action_lr = args.action_lr

        self.device = device

        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = (action_space.high - action_space.low) / 2.
            self.action_bias = (action_space.high + action_space.low) / 2.

    def append_memory(self, state, action, reward, next_state, mask):
        action = (action - self.action_bias) / self.action_scale

        self.memory.append(state, action, reward, next_state, mask)
        if not self.aug:
            self.diffusion_memory.append(state, action)

    def sample_action(self, state, eval=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        normal = False
        if not eval and torch.rand(1).item() <= self.epsilon:
            normal = True

        action = self.actor(state, eval, q_func=self.critic, normal=normal).cpu().data.numpy().flatten()
        action = action.clip(-1, 1)
        action = action * self.action_scale + self.action_bias
        return action

    def action_aug(self, batch_size, log_writer, return_mean_std=False):
        states, actions, rewards, next_states, masks = self.memory.sample(batch_size)
        old_states = states
        states, best_actions, v_target, (mean, std) = self.actor.sample_training(states,
                                                                                 action_samples=self.train_sample,
                                                                                 chosen=self.chosen,
                                                                                 q_func=self.critic) #  origin=actions
        v = v_target[1]

        if return_mean_std:
            return states, best_actions, (v_target[0], v), (mean, std)
        else:
            return states, best_actions, (v_target[0], v)

    def action_gradient(self, batch_size, log_writer, return_mean_std=False):
        states, best_actions, idxs = self.diffusion_memory.sample(batch_size)
        q1, q2 = self.critic(states, best_actions)
        q = torch.min(q1, q2)
        mean = q.mean()
        std = q.std()

        actions_optim = torch.optim.Adam([best_actions], lr=self.action_lr, eps=1e-5)

        for i in range(self.action_gradient_steps):
            best_actions.requires_grad_(True)
            q1, q2 = self.critic(states, best_actions)
            loss = -torch.min(q1, q2)

            actions_optim.zero_grad()

            loss.backward(torch.ones_like(loss))
            if self.action_grad_norm > 0:
                actions_grad_norms = nn.utils.clip_grad_norm_([best_actions], max_norm=self.action_grad_norm,
                                                              norm_type=2)

            actions_optim.step()

            best_actions.requires_grad_(False)
            best_actions.clamp_(-1., 1.)

        # if self.step % 10 == 0:
        #     log_writer.add_scalar('Action Grad Norm', actions_grad_norms.max().item(), self.step)

        best_actions = best_actions.detach()

        self.diffusion_memory.replace(idxs, best_actions.cpu().numpy())

        if return_mean_std:
            return states, best_actions, (mean, std)
        else:
            return states, best_actions

    def train(self, t, iterations, batch_size=256, log_writer=None):
        for itr in range(iterations):
            # Sample replay buffer / batch
            states, actions, rewards, next_states, masks = self.memory.sample(batch_size)

            """ Q Training """
            start_time = time.time()
            current_q1, current_q2 = self.critic(states, actions)

            next_actions = self.actor_target(next_states, eval=False, q_func=self.critic_target)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)

            target_q = (rewards + masks * target_q).detach()

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.ac_grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.ac_grad_norm,
                                                             norm_type=2)
                # if self.step % 10 == 0:
                #     log_writer.add_scalar('Critic Grad Norm', critic_grad_norms.max().item(), self.step)
            self.critic_optimizer.step()
            q_time = time.time()
            q_training_time = q_time - start_time

            """ Policy Training """
            if t % self.policy_freq == 0:
                if self.aug:
                    if self.gradient:
                        states, best_actions, qv, (mean, std) = self.aug_gradient(batch_size, log_writer, return_mean_std=True)
                    else:
                        # By default goes to here
                        states, best_actions, qv, (mean, std) = self.action_aug(batch_size, log_writer, return_mean_std=True)
                else:
                    states, best_actions, (mean, std) = self.action_gradient(batch_size, log_writer, return_mean_std=True)

                after_sample_time = time.time()
                action_sample_time = after_sample_time - q_time

                if self.policy_type == 'Diffusion' and self.weighted:
                    if self.aug:
                        q, v = qv
                    else:
                        v = None
                        with torch.no_grad():
                            q1, q2 = self.critic(states, best_actions)
                            q = torch.min(q1, q2)
                    # print("q shape", q.shape)
                    self.running_q_std += self.alpha_std * (std - self.running_q_std)
                    self.running_q_mean += self.alpha_mean * (mean - self.running_q_mean)
                    # q.clamp_(-self.q_neg).add_(self.q_neg)
                    q = eval(self.q_transform)(q, q_neg=self.q_neg, cut=self.cut, running_q_std=self.running_q_std, beta=self.beta,
                                               running_q_mean=self.running_q_mean, v=v, batch_size=batch_size, chosen=self.chosen)
                    if self.entropy_alpha > 0.0:
                        rand_states = states.unsqueeze(0).expand(10, -1, -1).contiguous().view(batch_size*self.chosen*10, -1)
                        rand_policy_actions = torch.empty(batch_size * self.chosen * 10, actions.shape[-1], device=self.device).uniform_(
                            -1, 1)
                        rand_q = q.unsqueeze(0).expand(10, -1, -1).contiguous().view(batch_size*self.chosen*10, -1) * self.entropy_alpha

                        best_actions = torch.cat([best_actions, rand_policy_actions], dim=0)
                        states = torch.cat([states, rand_states], dim=0)
                        q = torch.cat([q, rand_q], dim=0)
                    # q[q<1.0] = 1.0
                    # q = torch.clip(q / self.running_avg_qnorm, -6 ,6)
                    # expq = torch.exp(self.beta * q)
                    # expq[expq<=expq.quantile(0.95)] = 0.0
                    # if itr % 10000 == 0 : print(expq, itr)
                    # print("expq", expq.shape)
                    actor_loss = self.actor.loss(best_actions, states, weights=q)
                else:
                    actor_loss = self.actor.loss(best_actions, states)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                if self.ac_grad_norm > 0:
                    actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.ac_grad_norm,
                                                                norm_type=2)
                    # if self.step % 10 == 0:
                    #     log_writer.add_scalar('Actor Grad Norm', actor_grad_norms.max().item(), self.step)
                self.actor_optimizer.step()

            after_action_training_time = time.time()
            policy_training_time = after_action_training_time - after_sample_time

            """ Step Target network """
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            if self.step % self.update_actor_target_every == 0:
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

            if self.step % 50 == 0:
                log_writer.add_scalar('time/q_training_time', q_training_time, self.step)
                log_writer.add_scalar('time/action_sample_time', action_sample_time, self.step)
                log_writer.add_scalar('time/policy_training_time', policy_training_time, self.step)
                log_writer.add_scalar('stats/q_weights', q.mean().item(), self.step)
                log_writer.add_scalar('loss/actor_loss', actor_loss.item(), self.step)
                log_writer.add_scalar('loss/critic_loss', critic_loss.item(), self.step)
                log_writer.flush()

    def save_model(self, dir, id=None):
        if not os.path.exists(dir):
            os.mkdir(dir)
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')

        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')


    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth', map_location=self.device))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth', map_location=self.device))

        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth', map_location=self.device))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth', map_location=self.device))


    def aug_gradient(self, batch_size, log_writer, return_mean_std=False):

        states, best_actions, v, (mean, std) = self.action_aug(batch_size, log_writer, return_mean_std=True)


        actions_optim = torch.optim.Adam([best_actions], lr=self.action_lr, eps=1e-5)

        for i in range(self.action_gradient_steps):
            best_actions.requires_grad_(True)
            q1, q2 = self.critic(states, best_actions)
            loss = -torch.min(q1, q2)

            actions_optim.zero_grad()

            loss.backward(torch.ones_like(loss))
            if self.action_grad_norm > 0:
                actions_grad_norms = nn.utils.clip_grad_norm_([best_actions], max_norm=self.action_grad_norm,
                                                              norm_type=2)

            actions_optim.step()

            best_actions.requires_grad_(False)
            best_actions.clamp_(-1., 1.)

        # if self.step % 10 == 0:
        #     log_writer.add_scalar('Action Grad Norm', actions_grad_norms.max().item(), self.step)

        best_actions = best_actions.detach()

        _, v = v
        with torch.no_grad():
            q1, q2 = self.critic(states, best_actions)
            q = torch.min(q1, q2)

        if return_mean_std:
            return states, best_actions, (q, v), (mean, std)
        else:
            return states, best_actions, (q, v)

    def get_policy(self, states, times):
        batch_size = states.shape[0]
        states = states.unsqueeze(1).repeat(1, times, 1).view(times*batch_size, -1)
        actions = self.actor(states, eval=False, normal=True)
        return actions

    def get_value(self, states, actions):
        action_shape = actions.shape[0]
        state_shape = states.shape[0]
        rep = int(action_shape / state_shape)
        states = states.unsqueeze(1).repeat(1, rep, 1).view(rep*state_shape, -1)
        q1, q2 = self.critic(states, actions)
        return q1.view(state_shape, rep, 1), q2.view(state_shape, rep, 1)
    

class QVPOv2(QVPO):

    def __init__(self, args, state_dim, action_space, memory, diffusion_memory, device):
        super().__init__(args, state_dim, action_space, memory, diffusion_memory, device)
        self.args = args

    def train(self, t, iterations, batch_size=256, log_writer=None):
        for itr in range(iterations):
            # Sample replay buffer / batch
            states, actions, rewards, next_states, masks = self.memory.sample(batch_size)

            """ Q Training """
            start_time = time.time()
            current_q1, current_q2 = self.critic(states, actions)
            if self.args.use_action_target:
                next_actions = self.actor_target(next_states, eval=False, q_func=self.critic_target)
            else:
                next_actions = self.actor(next_states, eval=False, q_func=self.critic_target)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)

            target_q = (rewards + masks * target_q).detach()

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.ac_grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.ac_grad_norm,
                                                             norm_type=2)
                # if self.step % 10 == 0:
                #     log_writer.add_scalar('Critic Grad Norm', critic_grad_norms.max().item(), self.step)
            self.critic_optimizer.step()
            after_q_time = time.time()
            q_training_time = after_q_time - start_time

            """ Policy Training """
            if t % self.policy_freq == 0:
                # if self.action_sample == 'multi':
                states, best_actions, qv, (mean, std) = self.action_aug(batch_size, log_writer, return_mean_std=True)
                #
                after_sample_time = time.time()
                action_sample_time = after_sample_time - after_q_time
                #
                if self.policy_type == 'Diffusion' and self.weighted:
                    if self.aug:
                        q, v = qv
                    else:
                        v = None
                        with torch.no_grad():
                            q1, q2 = self.critic(states, best_actions)
                            q = torch.min(q1, q2)
                    # print("q shape", q.shape)
                #     self.running_q_std += self.alpha_std * (std - self.running_q_std)
                #     self.running_q_mean += self.alpha_mean * (mean - self.running_q_mean)

                # best_actions = self.actor(states, eval=False, normal=True)
                # after_sample_time = time.time()
                # action_sample_time = after_sample_time - after_q_time
                # q1, q2 = self.critic(states, best_actions)
                # q = torch.min(q1, q2)

                q_weights = eval(self.q_transform)(q)
                if self.entropy_alpha > 0.0:
                    rand_states = states.unsqueeze(0).expand(10, -1, -1).contiguous().view(batch_size*self.chosen*10, -1)
                    rand_policy_actions = torch.empty(batch_size * self.chosen * 10, actions.shape[-1], device=self.device).uniform_(
                        -1, 1)
                    rand_q = q_weights.unsqueeze(0).expand(10, -1, -1).contiguous().view(batch_size*self.chosen*10, -1) * self.entropy_alpha

                    best_actions = torch.cat([best_actions, rand_policy_actions], dim=0)
                    states = torch.cat([states, rand_states], dim=0)
                    q_weights = torch.cat([q_weights, rand_q], dim=0)
                actor_loss = self.actor.loss(best_actions, states, weights=q_weights)
                # else:
                #     actor_loss = self.actor.loss(best_actions, states)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                if self.ac_grad_norm > 0:
                    actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.ac_grad_norm,
                                                                norm_type=2)
                    # if self.step % 10 == 0:
                    #     log_writer.add_scalar('Actor Grad Norm', actor_grad_norms.max().item(), self.step)
                self.actor_optimizer.step()

            after_action_training_time = time.time()
            policy_training_time = after_action_training_time - after_q_time

            """ Step Target network """
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            if self.args.use_action_target:
                if self.step % self.update_actor_target_every == 0:
                    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

            if self.step % 50 == 0:
                log_writer.add_scalar('time/q_training_time', q_training_time, self.step)
                log_writer.add_scalar('time/action_sample_time', action_sample_time, self.step)
                log_writer.add_scalar('time/policy_training_time', policy_training_time, self.step)
                log_writer.add_scalar('stats/q_avg', q.mean().item(), self.step)
                log_writer.add_scalar('stats/q_max', q.max().item(), self.step)
                log_writer.add_scalar('stats/q_min', q.min().item(), self.step)
                log_writer.add_scalar('stats/q_weights_max', q_weights.max().item(), self.step)
                log_writer.add_scalar('loss/actor_loss', actor_loss.item(), self.step)
                log_writer.add_scalar('loss/critic_loss', critic_loss.item(), self.step)
                log_writer.flush()