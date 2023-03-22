import os
import torch
import wandb
import numpy as np
import torch.nn.functional as F
import imageio

from tqdm.auto import tqdm, trange
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import polyak_update

from af_guide.utils.env import create_env, evaluate_policy



class OnlineDTPolicyTrainer:
    def __init__(
            self,
            learning_rate=1e-4,
            betas=(0.9, 0.999),
            weight_decay=0.0,
            clip_grad=None,
            eval_seed=42,
            eval_every=10,
            eval_episodes=10,
            eval_sample_expand=1,
            eval_temperature=1,
            eval_discount=0.99,
            coe_env_r=1,
            coe_int_r=1,
            save_every=5,
            checkpoints_path=None,
            ablation_sac_reward_sum=False,  # ablation model, SAC with guided reward
            device="cpu",
    ):
        # optimizer params
        self.betas = betas
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clip_grad = clip_grad
        # loss params
        self.coe_env_r = coe_env_r
        self.coe_int_r = coe_int_r

        # eval params
        self.eval_seed = eval_seed
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes
        self.eval_sample_expand = eval_sample_expand
        self.eval_temperature = eval_temperature
        self.eval_discount = eval_discount

        # checkpoints
        self.save_every = save_every
        self.checkpoints_path = checkpoints_path

        self.ablation_sac_reward_sum = ablation_sac_reward_sum
        self.device = device

    def eval_explorer(self, env_name, explorer):
        vec_env = DummyVecEnv([lambda: create_env(env_name, eval=True) for _ in range(self.eval_episodes)])
        mean_reward, std_reward = evaluate_policy(explorer, vec_env, max_steps=vec_env.envs[0].max_episode_steps,)
        score = vec_env.envs[0].get_normalized_score(mean_reward)
        return mean_reward, std_reward, score


    def _batch_process_explorer(self, batch):
        states, actions, rewards_env, rewards_in, dones = batch
        # randomly select the first or second transition of the sampled sub episode as the training data.
        # simply select the first one will miss the final transition of each episodes
        # note that all the sub episodes are at least 2 steps long
        rand_idx = np.random.randint(2, size=len(states))
        rand_idx_next = rand_idx + 1
        mask = rand_idx[:, None] == np.arange(len(states[0]))
        mask_next = rand_idx_next[:, None] == np.arange(len(states[0]))

        states, next_states, actions, rewards_env, rewards_in, dones = \
            states[mask], states[mask_next], actions[mask], rewards_env[mask], rewards_in[mask], dones[mask]
        states, next_states, actions, rewards_env, rewards_in, dones = \
            [torch.tensor(i, device=self.device, dtype=torch.float)
             for i in [states, next_states, actions, rewards_env, rewards_in, dones]]
        return states, next_states, actions, rewards_env, rewards_in, dones

    def _get_loss_explorer(self, explorer, batch):
        states, next_states, actions, rewards_env, rewards_in, dones = batch

        if self.ablation_sac_reward_sum:
            rewards_env = self.coe_env_r * rewards_env + self.coe_int_r * rewards_in

        # ------------------------------------------
        # Action by the current actor for the sampled state
        actions_pi, log_prob = explorer.actor.action_log_prob(states)
        log_prob = log_prob.reshape(-1, 1)

        # ------------------------------------------
        # Compute entropy coefficient loss
        ent_coef_loss = None
        if explorer.ent_coef_optimizer is not None:
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            ent_coef = torch.exp(explorer.log_ent_coef.detach())
            ent_coef_loss = -(explorer.log_ent_coef * (log_prob + explorer.target_entropy).detach()).mean()
        else:
            ent_coef = explorer.ent_coef_tensor

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if ent_coef_loss is not None:
            explorer.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            explorer.ent_coef_optimizer.step()

        # ------------------------------------------
        # Compute critic_loss loss
        with torch.no_grad():
            # Select action according to policy
            next_actions, next_log_prob = explorer.actor.action_log_prob(next_states)
            # Compute the next Q values: min over all critics targets
            next_q_values = torch.cat(explorer.critic_target(next_states, next_actions), dim=1)
            next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
            # add entropy term
            next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
            # td error + entropy term
            target_q_values = rewards_env + (1 - dones) * explorer.gamma * next_q_values

        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        current_q_values = explorer.critic(states, actions)

        # Compute critic loss
        critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)

        # Optimize the critic
        explorer.critic.optimizer.zero_grad()
        critic_loss.backward()
        explorer.critic.optimizer.step()

        # ------------------------------------------
        # Compute critic_guided loss
        current_q_in = explorer.critic_guided(states, actions)[0]
        critic_guided_loss = F.mse_loss(current_q_in, rewards_in)

        # Optimize the critic_guided
        explorer.critic_guided.optimizer.zero_grad()
        critic_guided_loss.backward()
        explorer.critic_guided.optimizer.step()

        # ------------------------------------------
        # Compute actor loss
        # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
        # Min over all critic networks
        q_values_pi = torch.cat(explorer.critic.forward(states, actions_pi), dim=1)
        min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)

        if self.ablation_sac_reward_sum:
            q_pi = min_qf_pi
        else:
            q_in_pi = explorer.critic_guided.forward(states, actions_pi)[0]
            q_pi = self.coe_env_r * min_qf_pi + self.coe_int_r * q_in_pi

        actor_loss = (ent_coef * log_prob - q_pi).mean()

        # Optimize the actor
        explorer.actor.optimizer.zero_grad()
        actor_loss.backward()
        explorer.actor.optimizer.step()

        # Update target networks
        polyak_update(explorer.critic.parameters(), explorer.critic_target.parameters(), explorer.tau)

        return ent_coef, ent_coef_loss, critic_loss, actor_loss, critic_guided_loss

    def train_step_explorer(self, batch, explorer):
        # Switch to train mode (this affects batch norm / dropout)
        explorer.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [explorer.actor.optimizer, explorer.critic.optimizer]
        if explorer.ent_coef_optimizer is not None:
            optimizers += [explorer.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        explorer._update_learning_rate(optimizers)

        ent_coef, ent_coef_loss, critic_loss, actor_loss, critic_guided_loss = \
            self._get_loss_explorer(explorer, batch)
        explorer._n_updates += 1

        return ent_coef, ent_coef_loss, actor_loss, critic_loss, critic_guided_loss

    def train(self, env, combinator, replay_buffer, num_steps=100000, log_every=1000,
              init_env_steps=100, train_every=10, action_temperature=1, step_warm=0, int_reward_type='likelihood'):

        if self.checkpoints_path is not None:
            os.makedirs(self.checkpoints_path, exist_ok=True)

        data_collecter = OnlineDataCollecter(env,
                                             init_env_steps=init_env_steps,
                                             int_reward=int_reward_type,
                                             step_warm=step_warm)

        for total_steps in trange(0, num_steps, desc="Total Steps"):
            # environment step
            replay_buffer = data_collecter.collect_one_step(combinator, replay_buffer)
            combinator.n_steps = total_steps

            if total_steps < init_env_steps: continue
            # training step
            ent_coefs = []
            ent_coef_losses = []
            actor_losses = []
            critic_losses = []
            if total_steps % train_every == 0:
                batch, c_batch = replay_buffer.get_batch(batchsize=256, return_continuous=True)

                c_batch = self._batch_process_explorer(c_batch)
                ent_coef, ent_coef_loss, actor_loss, critic_loss, critic_guided_loss = \
                    self.train_step_explorer(c_batch, combinator.explorer)

                ent_coefs.append(ent_coef.item())
                ent_coef_losses.append(ent_coef_loss.item())
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

                ent_coef = np.mean(ent_coefs)
                ent_coef_loss = np.mean(ent_coef_losses)
                actor_loss = np.mean(actor_losses)
                critic_loss = np.mean(critic_losses)
                wandb.log({
                    "train/epoch": total_steps,
                    "train/explorer_ent_coef": ent_coef,
                    "train/explorer_ent_coef_loss": ent_coef_loss,
                    "train/explorer_actor_loss": actor_loss,
                    "train/explorer_critic_loss": critic_loss,
                    "train/explorer_critic_guided_loss": critic_guided_loss,
                })
                if total_steps % 100 == 0:
                    print(f'   Steps {total_steps}:', actor_loss)

            # evaluation
            if total_steps % self.eval_every == 0:
                reward_mean_explorer, reward_std_explorer, score_explorer = \
                    self.eval_explorer(env.envs[0].name, combinator.explorer)

                wandb.log({
                    "eval/explorer_reward_mean": reward_mean_explorer,
                    "eval/explorer_reward_std": reward_std_explorer,
                    "eval/explorer_score": score_explorer,
                    "eval/epoch": total_steps,
                })
                print(f"   EVAL {total_steps}:", reward_mean_explorer, reward_std_explorer)

            # save
            if self.checkpoints_path is not None and total_steps % self.save_every == 0:
                path = os.path.join(self.checkpoints_path, f"explorer_{total_steps}.pt")
                torch.save(combinator.explorer.policy.state_dict(), path)

        if self.checkpoints_path is not None:
            pass

        return combinator.explorer


class OnlineDataCollecter():
    def __init__(self, env, max_step=None, init_env_steps=1000, int_reward='l2', step_warm=0):
        self.env = env
        if max_step is None:
            self.max_step = self.env.envs[0].max_episode_steps
        else:
            self.max_step = max_step
        self.step_warm = step_warm
        self.total_step = 0
        self.obs = None
        self.done = True
        self.step = 0
        self.init_env_steps = init_env_steps
        assert int_reward in ['likelihood', 'l2', 'no'], 'Unknown intrinsic reward type {}'.format(int_reward)
        self.int_reward = int_reward
        # self.render_list = [] # for debug

    @property
    def cur_max_step(self):
        if self.step_warm == 0:
            return self.max_step
        else:
            cur_max_step = int(min(1, self.total_step / self.step_warm) * self.max_step)
            return max(cur_max_step, 1)

    def collect_one_step(self, model, replay_buffer):
        if self.done or self.step >= self.cur_max_step:
            self.obs = self.env.reset()[0]
            replay_buffer.new_episode()
            self.step = 0
            model.reset()
            # if len(self.render_list): # for debug
            #     imageio.mimsave("debug.gif", self.render_list, fps=64, format="gif")
            # self.render_list = [self.env.envs[0].render(mode="rgb_array", height=256, width=256)] # for debug

        goal, act_preds = model.sample_target_planner(self.obs)

        if len(replay_buffer) > self.init_env_steps:
            model.explorer.policy.set_training_mode(False)
            action = model.sample_action_explorer(self.obs)[0]
        else:
            action = self.env.action_space.sample()

        next_obs, reward, self.done, infos = self.env.step(action[None])

        if self.int_reward == 'l2':
            intrinsic_reward = -np.sqrt((((next_obs - goal) / model.planner.state_std.data.cpu().numpy()) ** 2).sum(-1))
        else:
            raise ValueError

        model.explorer.num_timesteps += 1
        model.explorer._update_current_progress_remaining(
            model.explorer.num_timesteps, model.explorer._total_timesteps)

        self.step = self.step + 1
        self.total_step = self.total_step + 1
        replay_buffer.append(self.obs, action, reward, intrinsic_reward, self.done)
        model.update_context_step(self.obs, action, reward)
        self.obs = next_obs[0]
        # self.render_list.append(self.env.envs[0].render(mode="rgb_array", height=256, width=256))  # for debug

        return replay_buffer
