import os
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np

from omegaconf import OmegaConf

import af_guide.env
from af_guide.utils.common import set_seed
from af_guide.utils.env import create_env
from af_guide.models.dt import DecisionTransformer, ActionFreeDecisionTransformer
from af_guide.models.sac import GuidedSAC, GuidedSACPolicy
from af_guide.models.combinator import CombinatorDT


def collect_one_step(env, model, max_steps):
    obs = env.reset()
    step = 0

    model.reset()
    model.explorer.policy.set_training_mode(False)

    targets = []
    reals = [obs]

    done = False
    while not done:
        goal, act_preds = model.sample_target_planner(obs)
        targets.append(goal)
        action = model.sample_action_explorer(obs)[0]
        env.set_state(goal[:2], goal[2:])
        next_obs = goal
        model.update_context_step(obs, action, np.array(0))

        # next_obs, reward, done, infos = env.step(action)
        # reals.append(next_obs)
        # done = done + (step >= max_steps)
        #
        # intrinsic_reward = -np.sqrt((((next_obs - goal) / model.planner.state_std.data.cpu().numpy()) ** 2).sum(-1))

        model.explorer.num_timesteps += 1
        step = step + 1
        # model.update_context_step(obs, action, np.array(reward))
        obs = next_obs

        env.render()

    targets = np.stack(targets)[:, :2]
    reals = np.stack(reals)[:, :2]
    plt.plot(reals[:, 0], reals[:, 1])
    for i in range(len(targets)):
        line_x = [reals[i, 0], targets[i, 0]]
        line_y = [reals[i, 1], targets[i, 1]]
        plt.plot(line_x, line_y, 'r')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.show()
    print('abc')




def create_argparser():
    parser = argparse.ArgumentParser(description="Trajectory Transformer evaluation hyperparameters. All can be set from command line.")
    parser.add_argument("--save_path", default="/home/zhud/ai/log/trajtran/checkpoints/maze2d-umaze-sparse-v1/uniform/dt_sac_acdt_int1_env1")
    parser.add_argument("--ckpt", default=100000, type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)

    return parser


def run_experiment(config, seed, device):
    set_seed(seed=seed)
    # env_name = config.dataset.env_name[:-3] + '-render' + config.dataset.env_name[-3:]
    env_name = config.dataset.env_name
    env = create_env(env_name)

    if config.model.acdt:
        DT = ActionFreeDecisionTransformer
    else:
        DT = DecisionTransformer

    planner = DT(state_dim=config.model.observation_dim,
                 act_dim=config.model.action_dim,
                 n_blocks=3,
                 h_dim=128,
                 context_len=20,
                 n_heads=1,
                 drop_p=0.1)

    planner.load_state_dict(torch.load(config.model.planner_path, map_location=device))
    planner.to(device)
    planner.eval()

    explorer = GuidedSAC(GuidedSACPolicy, env, device=device, verbose=1)

    explorer.policy.load_state_dict(torch.load(
        os.path.join(config.trainer.checkpoints_path, 'explorer_{}.pt'.format(config.ckpt))))

    combinator = CombinatorDT(planner=planner, explorer=explorer,
                              rtg_target=config.model.rtg_target, rtg_scale=config.model.rtg_scale)

    collect_one_step(env, combinator, env.max_episode_steps)

    print('finish')





def main():
    args, override = create_argparser().parse_known_args()
    config = OmegaConf.merge(
        OmegaConf.load(os.path.join(args.save_path, 'config.yaml')),
        OmegaConf.from_cli(override)
    )
    config.trainer.checkpoints_path = args.save_path
    config.ckpt = args.ckpt
    run_experiment(
        config=config,
        seed=args.seed,
        device=args.device
    )


if __name__ == "__main__":
    main()