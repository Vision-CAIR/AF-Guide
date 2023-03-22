import argparse
import os
import sys
import random
import csv
from datetime import datetime

import numpy as np
import gym
import d4rl

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from af_guide.utils.afdt import D4RLTrajectoryDataset, evaluate_on_env
from af_guide.models.dt import DecisionTransformer, ActionFreeDecisionTransformer, RewardActionFreeDecisionTransformer


def train(args):

    env_name = args.env
    env_d4rl_name = args.env
    if 'walker2d' in args.env:
        rtg_target = 5000
        rtg_scale = 1000

    elif 'halfcheetah' in args.env:
        rtg_target = 6000
        rtg_scale = 1000

    elif 'hopper' in args.env:
        rtg_target = 3600
        rtg_scale = 1000

    elif 'kitchen' in args.env:
        rtg_target = 350
        rtg_scale = 100

    elif 'antmaze' in args.env:
        rtg_target = 1
        rtg_scale = 1

    elif 'maze2d' in args.env:
        if 'umaze' in args.env:
            rtg_target = 150
            rtg_scale = 100
        else:
            rtg_target = 5000
            rtg_scale = 50

    elif 'pen' in args.env:
        rtg_target = 4000
        rtg_scale = 1000

    elif 'door' in args.env:
        rtg_target = 3500
        rtg_scale = 1000

    elif 'hammer' in args.env:
        rtg_target = 16000
        rtg_scale = 5000

    elif 'relocate' in args.env:
        rtg_target = 5000
        rtg_scale = 1000

    else:
        raise NotImplementedError

    max_eval_ep_len = args.max_eval_ep_len  # max len of one episode
    num_eval_ep = args.num_eval_ep          # num of evaluation episodes

    batch_size = args.batch_size            # training batch size
    lr = args.lr                            # learning rate
    wt_decay = args.wt_decay                # weight decay
    warmup_steps = args.warmup_steps        # warmup steps for lr scheduler

    # total updates = max_train_iters x num_updates_per_iter
    max_train_iters = args.max_train_iters
    num_updates_per_iter = args.num_updates_per_iter

    context_len = args.context_len      # K in decision transformer
    n_blocks = args.n_blocks            # num of transformer blocks
    embed_dim = args.embed_dim          # embedding (hidden) dim of transformer
    n_heads = args.n_heads              # num of transformer heads
    dropout_p = args.dropout_p          # dropout probability

    # load data from this file
    dataset_path = os.path.join(args.dataset_dir, env_d4rl_name + '.pkl')

    # saves model and csv in this directory
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # training and evaluation device
    device = torch.device(args.device)

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")

    if args.action_loss:
        prefix = "dt_" + env_d4rl_name
    else:
        prefix = "acdt_" + env_d4rl_name

    model_path = os.path.join(log_dir, prefix + "_model_" + start_time_str)
    os.makedirs(model_path)
    save_model_name = prefix + ".pt"
    save_model_path = os.path.join(model_path, save_model_name)
    save_best_model_path = save_model_path[:-3] + "_best.pt"

    log_csv_name = prefix + "_log_" + start_time_str + ".csv"
    log_csv_path = os.path.join(model_path, log_csv_name)

    csv_writer = csv.writer(open(log_csv_path, 'a', 1))
    csv_header = (["duration", "num_updates", "action_loss", "state_loss",
                   "valid_action_loss", "valid_state_loss",
                   "eval_avg_reward", "eval_avg_ep_len", "eval_d4rl_score"])

    csv_writer.writerow(csv_header)

    print("=" * 60)
    print("start time: " + start_time_str)
    print("=" * 60)

    print("device set to: " + str(device))
    print("dataset path: " + dataset_path)
    print("model save path: " + save_model_path)
    print("log csv save path: " + log_csv_path)

    traj_dataset = D4RLTrajectoryDataset(dataset_path, context_len, rtg_scale)
    ## get state stats from dataset
    state_stats = traj_dataset.get_state_stats()
    state_mean, state_std, diff_state_mean, diff_state_std = \
        [torch.tensor(stats, requires_grad=False).to(device) for stats in state_stats]

    train_size = int(len(traj_dataset) // 10 * 9)
    valid_size = len(traj_dataset) - train_size
    traj_dataset, traj_dataset_valid = torch.utils.data.random_split(traj_dataset, [train_size, valid_size])

    traj_data_loader = DataLoader(
        traj_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False
    )
    traj_data_valid_loader = DataLoader(
        traj_dataset_valid,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False
    )

    data_iter = iter(traj_data_loader)

    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    if args.action_loss:
        ModelClass = DecisionTransformer
    else:
        ModelClass = ActionFreeDecisionTransformer

    model = ModelClass(
        state_dim=state_dim,
        act_dim=act_dim,
        n_blocks=n_blocks,
        h_dim=embed_dim,
        context_len=context_len,
        n_heads=n_heads,
        drop_p=dropout_p,
    )

    model.load_state_stats(state_mean.cpu().numpy(), state_std.cpu().numpy(),
                           diff_state_mean.cpu().numpy(), diff_state_std.cpu().numpy())
    model.to(device)

    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=lr,
                        weight_decay=wt_decay
                    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
                            optimizer,
                            lambda steps: min((steps+1)/warmup_steps, 1)
                        )

    max_d4rl_score = -1.0
    total_updates = 0

    for i_train_iter in range(max_train_iters):

        log_action_losses = []
        log_state_losses = []
        model.train()

        for _ in range(num_updates_per_iter):
            try:
                timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)
            except StopIteration:
                data_iter = iter(traj_data_loader)
                timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)

            timesteps = timesteps.to(device)    # B x T
            states = states.to(device)          # B x T x state_dim
            actions = actions.to(device)        # B x T x act_dim
            returns_to_go = returns_to_go.to(device).unsqueeze(dim=-1).type(torch.float32) # B x T x 1
            traj_mask = traj_mask.to(device)    # B x T
            action_target = torch.clone(actions).detach().to(device)
            diff_state_target = (states[:, 1:] - states[:, :-1]) * state_std / diff_state_std - diff_state_mean

            diff_state_preds, action_preds, return_preds = model.forward(
                                                            timesteps=timesteps,
                                                            states=states,
                                                            actions=actions,
                                                            returns_to_go=returns_to_go
                                                        )
            # only consider non padded elements
            state_mask = (traj_mask[:, :-1] > 0) * (traj_mask[:, 1:] > 0)
            diff_state_preds = diff_state_preds[:, :-1].reshape(-1, state_dim)[state_mask.view(-1, )]
            diff_state_target = diff_state_target.reshape(-1, state_dim)[state_mask.view(-1, )]

            action_preds = action_preds.view(-1, act_dim)[traj_mask.view(-1,) > 0]
            action_target = action_target.view(-1, act_dim)[traj_mask.view(-1,) > 0]

            action_loss = F.l1_loss(action_preds, action_target, reduction='mean') * int(args.action_loss)  # afdt doesn't train to predict action
            state_loss = F.l1_loss(diff_state_preds, diff_state_target, reduction='mean')

            optimizer.zero_grad()
            (action_loss + state_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            scheduler.step()

            log_action_losses.append(action_loss.detach().cpu().item())
            log_state_losses.append(state_loss.detach().cpu().item())

        # evaluate action accuracy. useless for afdt
        results = evaluate_on_env(model, device, context_len, env, rtg_target, rtg_scale,
                                  num_eval_ep, max_eval_ep_len,
                                  state_mean, state_std,
                                  diff_state_mean, diff_state_std)

        valid_action_loss, valid_state_loss = eval_valid(model, traj_data_valid_loader, device,
                                                         state_std, diff_state_std, diff_state_mean)

        eval_avg_reward = results['eval/avg_reward']
        eval_avg_ep_len = results['eval/avg_ep_len']
        eval_d4rl_score = env.get_normalized_score(results['eval/avg_reward']) * 100

        mean_action_loss = np.mean(log_action_losses)
        mean_state_loss = np.mean(log_state_losses)
        time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)

        total_updates += num_updates_per_iter

        log_str = ("=" * 60 + '\n' +
                   "time elapsed: " + time_elapsed + '\n' +
                   "num of updates: " + str(total_updates) + '\n' +
                   "action loss: " + format(mean_action_loss, ".5f") + '\n' +
                   "state loss: " + format(mean_state_loss, ".5f") + '\n' +
                   "valid action loss: " + format(valid_action_loss, ".5f") + '\n' +
                   "valid state loss: " + format(valid_state_loss, ".5f") + '\n' +
                   "eval avg reward: " + format(eval_avg_reward, ".5f") + '\n' +
                   "eval avg ep len: " + format(eval_avg_ep_len, ".5f") + '\n' +
                   "eval d4rl score: " + format(eval_d4rl_score, ".5f")
                   )

        print(log_str)

        log_data = [time_elapsed, total_updates, mean_action_loss, mean_state_loss,
                    valid_action_loss, valid_state_loss,
                    eval_avg_reward, eval_avg_ep_len,
                    eval_d4rl_score]

        csv_writer.writerow(log_data)

        # save model
        print("max d4rl score: " + format(max_d4rl_score, ".5f"))
        if eval_d4rl_score >= max_d4rl_score:
            print("saving max d4rl score model at: " + save_best_model_path)
            torch.save(model.state_dict(), save_best_model_path)
            max_d4rl_score = eval_d4rl_score

        if total_updates % 1000 == 0:
            save_path_with_ckpt = save_model_path[:-3] + "_{}.pt".format(total_updates)
            print("saving current model at: " + save_path_with_ckpt)
            torch.save(model.state_dict(), save_path_with_ckpt)


    print("=" * 60)
    print("finished training!")
    print("=" * 60)
    end_time = datetime.now().replace(microsecond=0)
    time_elapsed = str(end_time - start_time)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    print("started training at: " + start_time_str)
    print("finished training at: " + end_time_str)
    print("total training time: " + time_elapsed)
    print("max d4rl score: " + format(max_d4rl_score, ".5f"))
    print("saved max d4rl score model at: " + save_best_model_path)
    print("saved last updated model at: " + save_model_path)
    print("=" * 60)


@torch.no_grad()
def eval_valid(model, data_iter, device, state_std, diff_state_std, diff_state_mean):
    model.eval()
    log_action_losses = []
    log_state_losses = []
    for timesteps, states, actions, returns_to_go, traj_mask in data_iter:
        timesteps = timesteps.to(device)  # B x T
        states = states.to(device)  # B x T x state_dim
        actions = actions.to(device)  # B x T x act_dim
        returns_to_go = returns_to_go.to(device).unsqueeze(dim=-1).type(torch.float32)  # B x T x 1
        traj_mask = traj_mask.to(device)  # B x T
        action_target = torch.clone(actions).detach().to(device)
        diff_state_target = (states[:, 1:] - states[:, :-1]) * state_std / diff_state_std - diff_state_mean

        diff_state_preds, action_preds, return_preds = model.forward(
            timesteps=timesteps,
            states=states,
            actions=actions,
            returns_to_go=returns_to_go
        )
        # only consider non padded elements
        state_mask = (traj_mask[:, :-1] > 0) * (traj_mask[:, 1:] > 0)
        diff_state_preds = diff_state_preds[:, :-1].reshape(-1, model.state_dim)[state_mask.view(-1, )]
        diff_state_target = diff_state_target.reshape(-1, model.state_dim)[state_mask.view(-1, )]

        action_preds = action_preds.view(-1, model.act_dim)[traj_mask.view(-1, ) > 0]
        action_target = action_target.view(-1, model.act_dim)[traj_mask.view(-1, ) > 0]

        action_loss = F.l1_loss(action_preds, action_target, reduction='mean')
        state_loss = F.l1_loss(diff_state_preds, diff_state_target, reduction='mean')

        log_action_losses.append(action_loss.detach().cpu().item())
        log_state_losses.append(state_loss.detach().cpu().item())
    mean_action_loss = np.mean(log_action_losses)
    mean_state_loss = np.mean(log_state_losses)
    model.train()
    return mean_action_loss, mean_state_loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='kitchen-complete-v0')

    parser.add_argument('--max_eval_ep_len', type=int, default=1000)
    parser.add_argument('--num_eval_ep', type=int, default=10)
    parser.add_argument('--action_loss', action='store_true')

    parser.add_argument('--dataset_dir', type=str, default='afdt_data/')
    parser.add_argument('--log_dir', type=str, default='dt_runs/')

    parser.add_argument('--context_len', type=int, default=20)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--dropout_p', type=float, default=0.1)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wt_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)

    parser.add_argument('--max_train_iters', type=int, default=100)  # 30 60 120 medium, medium expert, medium replay 10 maze2d 60 antmaze umaze
    parser.add_argument('--num_updates_per_iter', type=int, default=500)

    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    train(args)
