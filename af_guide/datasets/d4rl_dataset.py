import os
import torch
import numpy as np
import random

import pickle
from tqdm.auto import trange, tqdm
from torch.utils.data import Dataset

from af_guide.datasets.get_d4rl import d4rl_dataset
from af_guide.utils.common import pad_along_axis
from af_guide.utils.discretization import KBinsDiscretizer
from af_guide.utils.env import create_env


def join_trajectory(states, actions, rewards, discount=0.99):
    traj_length = states.shape[0]
    # I can vectorize this for all dataset as once,
    # but better to be safe and do it once and slow and right (and cache it)
    discounts = (discount ** np.arange(traj_length))

    values = np.zeros_like(rewards)
    for t in range(traj_length):
        # discounted return-to-go from state s_t:
        # r_{t+1} + y * r_{t+2} + y^2 * r_{t+3} + ...
        values[t] = (rewards[t + 1:] * discounts[:-t - 1]).sum()

    joined_transition = np.concatenate([states, actions, rewards, values], axis=-1)

    return joined_transition


def segment(states, actions, rewards, terminals):
    assert len(states) == len(terminals)
    trajectories = {}

    episode_num = 0
    for t in trange(len(terminals), desc="Segmenting"):
        if episode_num not in trajectories:
            trajectories[episode_num] = {
                "states": [],
                "actions": [],
                "rewards": []
            }
        
        trajectories[episode_num]["states"].append(states[t])
        trajectories[episode_num]["actions"].append(actions[t])
        trajectories[episode_num]["rewards"].append(rewards[t])

        if terminals[t].item():
            # next episode
            episode_num = episode_num + 1

    trajectories_lens = [len(v["states"]) for k, v in trajectories.items()]

    for t in trajectories:
        trajectories[t]["states"] = np.stack(trajectories[t]["states"], axis=0)
        trajectories[t]["actions"] = np.stack(trajectories[t]["actions"], axis=0)
        trajectories[t]["rewards"] = np.stack(trajectories[t]["rewards"], axis=0)

    return trajectories, trajectories_lens


# adapted from https://github.com/jannerm/trajectory-transformer/blob/master/trajectory/datasets/sequence.py
class DiscretizedDataset(Dataset):
    def __init__(self, env_name, num_bins=100, seq_len=10, discount=0.99,
                 strategy="uniform", cache_path=None, mode='standard', scale=1.0):
        self.seq_len = seq_len
        self.discount = discount
        self.num_bins = num_bins
        self.env = create_env(env_name)
        self.mode = mode

        dataset = d4rl_dataset(self.env)
        trajectories, traj_lengths = segment(
            dataset["states"],
            dataset["actions"],
            dataset["rewards"],
            dataset["dones"]
        )
        self.state_dim = dataset['states'].shape[-1]
        self.action_dim = dataset['actions'].shape[-1]

        self.cache_path = cache_path
        self.cache_name = f"{env_name}_{num_bins}_{seq_len}_{strategy}_{discount}"
        if scale < 1.0:
            full_cache = self.cache_name
            self.cache_name = self.cache_name + '_{}'.format(scale)

            num_traj = len(trajectories)
            select_idx = random.sample(range(num_traj), int(scale * num_traj))

            trajectories = {i: trajectories[i] for i in select_idx}
            traj_lengths = [traj_lengths[i] for i in select_idx]

        if cache_path is None or not os.path.exists(os.path.join(cache_path, self.cache_name)):
            self.joined_transitions = []
            for t in tqdm(trajectories, desc="Joining transitions"):
                self.joined_transitions.append(
                    join_trajectory(trajectories[t]["states"], trajectories[t]["actions"], trajectories[t]["rewards"])
                )

            os.makedirs(os.path.join(cache_path), exist_ok=True)
            # save cached version
            with open(os.path.join(cache_path, self.cache_name), "wb") as f:
                pickle.dump(self.joined_transitions, f)
        else:
            with open(os.path.join(cache_path, self.cache_name), "rb") as f:
                self.joined_transitions = pickle.load(f)

        if scale == 1:
            self.discretizer = KBinsDiscretizer(
                np.concatenate(self.joined_transitions, axis=0),
                num_bins=num_bins,
                strategy=strategy
            )
        else:
            with open(os.path.join(cache_path, full_cache), "rb") as f:
                full_joined_transitions = pickle.load(f)
            self.discretizer = KBinsDiscretizer(
                np.concatenate(full_joined_transitions, axis=0),
                num_bins=num_bins,
                strategy=strategy
            )

        # get valid indices for seq_len sampling
        indices = []
        for path_ind, length in enumerate(traj_lengths):
            end = length - 1
            for i in range(end):
                indices.append((path_ind, i, i + self.seq_len))
        self.indices = np.array(indices)

    def get_env_name(self):
        return self.env.name

    def get_discretizer(self):
        return self.discretizer

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_idx, start_idx, end_idx = self.indices[idx]
        joined = self.joined_transitions[traj_idx][start_idx:end_idx]
        if self.mode == 'policy':
            joined = joined[:, :self.state_dim + self.action_dim]
        elif self.mode == 'planner':
            joined = np.concatenate([joined[:, :self.state_dim],
                                     joined[:, self.state_dim + self.action_dim:]], axis=1)

        loss_pad_mask = np.ones((self.seq_len, joined.shape[-1]))
        if joined.shape[0] < self.seq_len:
            # pad to seq_len if at the end of trajectory, mask for padding
            loss_pad_mask[joined.shape[0]:] = 0
            joined = pad_along_axis(joined, pad_to=self.seq_len, axis=0)

        if self.mode == 'planner':
            indices = list(range(self.state_dim)) + \
                      list(range(self.state_dim + self.action_dim, self.state_dim + self.action_dim + 2))
            joined_discrete = self.discretizer.encode(joined, indices=indices).reshape(-1).astype(np.long)
        else:
            joined_discrete = self.discretizer.encode(joined).reshape(-1).astype(np.long)
        loss_pad_mask = loss_pad_mask.reshape(-1)

        return joined_discrete[:-1], joined_discrete[1:], loss_pad_mask[:-1]


class DiscretizedReplayBuffer(Dataset):
    # replay buffer version of the discretized dataset for online training/tuning
    def __init__(self, discretizer, state_dim, action_dim, max_episode_len=1000,
                 seq_len=10, discount=0.99, mode='no_r'):
        self.seq_len = seq_len
        self.discount = discount
        self.discretizer = discretizer
        self.mode = mode

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_episode_len = max_episode_len

        self.episode_pointer = -1
        self.step_pointer = 0
        self.joined_transitions = []
        self.indices = []

    def new_episode(self):
        if self.episode_pointer != -1:
            if self.step_pointer == 0:  # previous episode is empty, not need for adding a new one
                return
            # remove the padded part of current episode
            self.joined_transitions[-1] = self.joined_transitions[-1][:self.step_pointer]

        # create an episode placeholder
        empty_episode = np.zeros([self.max_episode_len, self.state_dim + self.action_dim + 3])
        self.joined_transitions.append(empty_episode)

        # set pointers
        self.episode_pointer = len(self.joined_transitions) - 1
        self.step_pointer = 0

    def append(self, state, action, reward, int_reward, done):
        # add new data to the buffer by change the placeholder in inplace way (inplace for speed reason)
        self.joined_transitions[self.episode_pointer][self.step_pointer] = \
            np.concatenate([state, action, reward, int_reward, done])
        # update indices when the episode is at least 2 steps long to contain new data
        if self.step_pointer >= 1:
            self.indices.append((self.episode_pointer, self.step_pointer - 1, self.step_pointer + self.seq_len - 1))
        self.step_pointer = self.step_pointer + 1

    def verify_index(self, traj_idx, start_idx, end_idx):
        # check if the fetched data will contain empty steps in episode placeholder. truncate it if yes
        if traj_idx == self.episode_pointer:
            end_idx = min(end_idx, self.step_pointer)
        return traj_idx, start_idx, end_idx

    def get_discretizer(self):
        return self.discretizer

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_idx, start_idx, end_idx = self.verify_index(*self.indices[idx])
        joined = self.joined_transitions[traj_idx][start_idx:end_idx]

        loss_pad_mask = np.ones((self.seq_len, joined.shape[-1]))
        if joined.shape[0] < self.seq_len:
            # pad to seq_len if at the end of trajectory, mask for padding
            loss_pad_mask[joined.shape[0]:] = 0
            joined = pad_along_axis(joined, pad_to=self.seq_len, axis=0)

        return joined, loss_pad_mask

    def get_batch(self, batchsize, return_continuous=False, indices=None):
        if indices is None:
            indices = np.random.randint(0, len(self), batchsize)
        batch = []
        continuous_batch = []

        for i in indices:
            joined, loss_pad_mask = self[i]
            if return_continuous:
                states = joined[:, :self.state_dim]
                actions = joined[:, self.state_dim:self.state_dim+self.action_dim]
                reward = joined[:, self.state_dim+self.action_dim:-2]
                intrinsic_reward = joined[:, -2:-1]
                dones = 1 - loss_pad_mask[:, :1] + joined[:, -1:]
                continuous_batch.append([states, actions, reward, intrinsic_reward, dones])

            if not self.discretizer is None:
                joined = joined[:, :-3]  # remove reward, intrinsic reward and done. do not use here
                loss_pad_mask = loss_pad_mask[:, :-3]
                joined_discrete = self.discretizer.encode(joined).reshape(-1).astype(np.long)
                loss_pad_mask = loss_pad_mask.reshape(-1)
                batch.append([joined_discrete[:-1], joined_discrete[1:], loss_pad_mask[:-1]])

        if not self.discretizer is None:
            batch = [np.stack(i) for i in zip(*batch)]

        if return_continuous:
            continuous_batch = [np.stack(i) for i in zip(*continuous_batch)]
            return batch, continuous_batch
        else:
            return batch

