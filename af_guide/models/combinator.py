import numpy as np
import torch


class CombinatorDT():
    def __init__(self, planner, explorer, rtg_target, rtg_scale, max_steps=1000, num_envs=1):
        self.planner = planner
        self.explorer = explorer
        self.max_steps = max_steps
        self.device = list(self.planner.parameters())[0].device
        self.num_envs = num_envs

        self.obs_dim = planner.state_dim
        self.act_dim = planner.act_dim

        self.step = -1

        timesteps = torch.arange(start=0, end=self.max_steps, step=1)
        self.timesteps = timesteps.repeat(self.num_envs, 1).to(self.device)

        self.rtg_target = torch.ones(self.num_envs, 1, device=self.device) * rtg_target
        self.rtg_scale = rtg_scale

    def reset(self, num_envs=None):
        if not num_envs is None:
            self.num_envs = num_envs

        self.actions = torch.zeros((self.num_envs, self.max_steps, self.act_dim),
                              dtype=torch.float32, device=self.device)
        self.states = torch.zeros((self.num_envs, self.max_steps, self.obs_dim),
                             dtype=torch.float32, device=self.device)
        self.rewards_to_go = torch.zeros((self.num_envs, self.max_steps, 1),
                                    dtype=torch.float32, device=self.device)

        self.step = 0
        self.running_rtg = self.rtg_target / self.rtg_scale

    def update_context_step(self, obs, action, reward):
        obs = torch.from_numpy(obs).to(self.device)
        obs = (obs - self.planner.state_mean) / self.planner.state_std
        if len(obs.shape) == 1:
            obs = obs[None]
        action = torch.from_numpy(action).to(self.device)
        if len(action.shape) == 1:
            action = action[None]
        reward = torch.from_numpy(reward).to(self.device)
        if len(reward.shape) == 1:
            reward = reward[None]

        self.states[:, self.step] = obs
        self.actions[:, self.step] = action
        self.rewards_to_go[:, self.step] = self.running_rtg
        self.running_rtg = self.running_rtg - (reward / self.rtg_scale)
        self.step = self.step + 1

    def sample_target_planner(self, obs):
        orig_obs = obs
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        obs = (obs - self.planner.state_mean) / self.planner.state_std
        if len(obs.shape) == 1:
            obs = obs[None]

        context_begin = max(0, self.step + 1 - self.planner.context_len)

        timesteps = self.timesteps[:, context_begin:self.step+1]
        states = self.states[:, context_begin:self.step]
        actions = self.actions[:, context_begin:self.step+1]
        rewards_to_go = self.rewards_to_go[:, context_begin:self.step]

        states = torch.cat([states, obs[:, None]], dim=1)
        rewards_to_go = torch.cat([rewards_to_go, self.running_rtg[:, None]], dim=1)

        diff_state_preds, act_preds, _ = self.planner(timesteps, states, actions, rewards_to_go)
        diff_state_preds = diff_state_preds[0, -1].detach()
        diff_state_preds = diff_state_preds * self.planner.diff_state_std + self.planner.diff_state_mean
        goal = orig_obs + diff_state_preds.cpu().numpy()

        act_preds = act_preds[0, -1].detach().cpu().numpy()

        return goal, act_preds

    def sample_action_explorer(self, obs):
        unscaled_action, _ = self.explorer.predict(obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        scaled_action = self.explorer.policy.scale_action(unscaled_action)

        # We store the scaled action in the buffer
        buffer_action = scaled_action
        action = self.explorer.policy.unscale_action(scaled_action)

        return action, buffer_action