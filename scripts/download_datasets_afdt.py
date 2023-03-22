import os
import gym
import numpy as np

import collections
import pickle

import d4rl

dataset_names = [
	'maze2d-medium-v1',
	'maze2d-large-v1',
	'antmaze-umaze-v0',
	'antmaze-umaze-diverse-v0',
	'walker2d-medium-v2',
	'walker2d-medium-expert-v2',
	'walker2d-medium-replay-v2',
	'halfcheetah-medium-v2',
	'halfcheetah-medium-expert-v2',
	'halfcheetah-medium-replay-v2',
	'hopper-medium-v2',
	'hopper-medium-expert-v2',
	'hopper-medium-replay-v2'
]


def download_d4rl_data():
	datasets = []

	data_dir = 'afdt_data/'

	print(data_dir)

	if not os.path.exists(data_dir):
		os.makedirs(data_dir)

	for name in dataset_names:

		pkl_file_path = os.path.join(data_dir, name)

		print("processing: ", name)

		env = gym.make(name)
		dataset = env.get_dataset()

		N = dataset['rewards'].shape[0]
		data_ = collections.defaultdict(list)

		use_timeouts = False
		if 'timeouts' in dataset:
			use_timeouts = True

		episode_step = 0
		paths = []
		for i in range(N):
			done_bool = bool(dataset['terminals'][i])
			if use_timeouts:
				final_timestep = dataset['timeouts'][i]
			else:
				final_timestep = (episode_step == 1000 - 1)
			for k in ['observations', 'actions', 'rewards', 'terminals']:
				data_[k].append(dataset[k][i])
			if done_bool or final_timestep:
				episode_step = 0
				episode_data = {}
				for k in data_:
					episode_data[k] = np.array(data_[k])
				paths.append(episode_data)
				data_ = collections.defaultdict(list)
			episode_step += 1

		returns = np.array([np.sum(p['rewards']) for p in paths])
		num_samples = np.sum([p['rewards'].shape[0] for p in paths])
		print(f'Number of samples collected: {num_samples}')
		print(
			f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

		with open(f'{pkl_file_path}.pkl', 'wb') as f:
			pickle.dump(paths, f)


if __name__ == "__main__":
	download_d4rl_data()
