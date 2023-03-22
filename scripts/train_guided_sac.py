import os
import wandb
import torch
import argparse
import numpy as np
import gym
import d4rl

from tqdm.auto import trange
from omegaconf import OmegaConf

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import configure_logger

import af_guide.env
from af_guide.models.dt import DecisionTransformer, ActionFreeDecisionTransformer, RewardActionFreeDecisionTransformer
from af_guide.models.sac import GuidedSAC, GuidedSACPolicy
from af_guide.models.combinator import CombinatorDT
from af_guide.models.dt.dt_trainer import OnlineDTPolicyTrainer
from af_guide.datasets.d4rl_dataset import DiscretizedReplayBuffer
from af_guide.utils.common import set_seed, get_parameters_by_name
from af_guide.utils.env import create_env


def create_argparser():
    parser = argparse.ArgumentParser(description="Trajectory Transformer evaluation hyperparameters. All can be set from command line.")
    parser.add_argument("--config", default="configs/guided_sac/maze2d_medium.yaml")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--ckpt_root", default='./', type=str)
    parser.add_argument("--planner_path", type=str, default='', help='if not specified, the default planner path in the config file will be used.')
    parser.add_argument("--name_postfix", type=str, required=True, help='used to specify the name postfix of the save folder')
    parser.add_argument("--ablation_sac_reward_sum", action='store_true', help='ablation, sac with intrinsic reward')

    return parser


def run_experiment(config, seed, device):
    env = DummyVecEnv([lambda: create_env(config.dataset.env_name) for _ in range(1)])

    if config.model.type == 'afdt':
        DT = ActionFreeDecisionTransformer
    elif config.model.type == 'rafdt':
        DT = RewardActionFreeDecisionTransformer
    elif config.model.type == 'dt':
        DT = DecisionTransformer
    else:
        raise NotImplemented

    planner = DT(state_dim=config.model.observation_dim,
                 act_dim=config.model.action_dim,
                 n_blocks=config.model.num_layers,
                 h_dim=config.model.embedding_dim,
                 context_len=config.model.context_len,
                 n_heads=config.model.num_heads,
                 drop_p=0.1)

    planner.load_state_dict(torch.load(config.model.planner_path, map_location=device))
    planner.to(device)
    planner.eval()

    trainer_conf = config.trainer
    if trainer_conf.checkpoints_path is not None:
        os.makedirs(trainer_conf.checkpoints_path, exist_ok=True)
        OmegaConf.save(OmegaConf.to_container(config, resolve=True),
                       os.path.join(trainer_conf.checkpoints_path, "config.yaml"))

    set_seed(seed=seed)

    # initialize training model
    explorer = GuidedSAC(GuidedSACPolicy, env, device=device, verbose=1)

    explorer.set_logger(configure_logger())
    explorer._total_timesteps = trainer_conf.num_steps
    explorer.batch_norm_stats = get_parameters_by_name(explorer.critic, ["running_"])
    explorer.batch_norm_stats_target = get_parameters_by_name(explorer.critic_target, ["running_"])

    combinator = CombinatorDT(planner=planner, explorer=explorer,
                              rtg_target=config.model.rtg_target, rtg_scale=config.model.rtg_scale)

    replay_buffer = DiscretizedReplayBuffer(None,
                                            config.model.observation_dim,
                                            config.model.action_dim,
                                            max_episode_len=env.envs[0].max_episode_steps)

    wandb.init(
        **config.wandb,
        config=dict(OmegaConf.to_container(config, resolve=True))
    )

    trainer = OnlineDTPolicyTrainer(
        coe_env_r=trainer_conf.coe_env_r,
        coe_int_r=trainer_conf.coe_int_r,
        learning_rate=trainer_conf.lr,
        betas=trainer_conf.betas,
        weight_decay=trainer_conf.weight_decay,
        clip_grad=trainer_conf.clip_grad,
        eval_seed=trainer_conf.eval_seed,
        eval_every=trainer_conf.eval_every,
        eval_episodes=trainer_conf.eval_episodes,
        checkpoints_path=trainer_conf.checkpoints_path,
        save_every=100000,
        device=device,
        ablation_sac_reward_sum=config.model.ablation_sac_reward_sum
    )
    trainer.train(
        env=env,
        combinator=combinator,
        replay_buffer=replay_buffer,
        train_every=trainer_conf.train_every,
        num_steps=trainer_conf.num_steps,
        init_env_steps=100,
        int_reward_type='l2',
        step_warm=trainer_conf.step_warm
    )


def main():
    args, override = create_argparser().parse_known_args()
    config = OmegaConf.merge(
        OmegaConf.load(args.config),
        OmegaConf.from_cli(override)
    )

    if args.planner_path:
        config.model.planner_path = args.planner_path
    save_name = config.trainer.name_prefix + '_' + args.name_postfix
    config.trainer.checkpoints_path = os.path.join(args.ckpt_root, config.trainer.checkpoints_root, save_name)
    config.wandb.name = save_name
    config.model.ablation_sac_reward_sum = args.ablation_sac_reward_sum

    run_experiment(
        config=config,
        seed=args.seed,
        device=args.device
    )


if __name__ == "__main__":
    main()
