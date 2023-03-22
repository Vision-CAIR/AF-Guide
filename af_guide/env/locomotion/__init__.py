from gym.envs.registration import register
from d4rl.locomotion import maze_env

"""
register(
    id='antmaze-umaze-render-v0',
    entry_point='trajectory.env.locomotion.ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'maze_map': maze_env.U_MAZE_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse.hdf5',
        'non_zero_reset':False,
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)
"""

register(
    id='antmaze-umaze-render-v0',
    entry_point='trajectory.env.locomotion.ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'maze_map': maze_env.U_MAZE_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse.hdf5',
        'non_zero_reset':False,
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-umaze-diverse-render-v0',
    entry_point='trajectory.env.locomotion.ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'maze_map': maze_env.U_MAZE_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_True_multigoal_True_sparse.hdf5',
        'non_zero_reset':False,
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-medium-play-render-v0',
    entry_point='trajectory.env.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.BIG_MAZE_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_big-maze_noisy_multistart_True_multigoal_False_sparse.hdf5',
        'non_zero_reset':False,
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-medium-diverse-render-v0',
    entry_point='trajectory.env.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.BIG_MAZE_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_big-maze_noisy_multistart_True_multigoal_True_sparse.hdf5',
        'non_zero_reset':False,
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-large-diverse-render-v0',
    entry_point='trajectory.env.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.HARDEST_MAZE_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_True_sparse.hdf5',
        'non_zero_reset':False,
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-large-play-render-v0',
    entry_point='trajectory.env.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.HARDEST_MAZE_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse.hdf5',
        'non_zero_reset':False,
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-umaze-render-v1',
    entry_point='trajectory.env.locomotion.ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'maze_map': maze_env.U_MAZE_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v1/Ant_maze_umaze_noisy_multistart_False_multigoal_False_sparse.hdf5',
        'non_zero_reset':False,
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-umaze-diverse-render-v1',
    entry_point='trajectory.env.locomotion.ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'maze_map': maze_env.U_MAZE_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v1/Ant_maze_umaze_noisy_multistart_True_multigoal_True_sparse.hdf5',
        'non_zero_reset':False,
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-medium-play-render-v1',
    entry_point='trajectory.env.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.BIG_MAZE_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v1/Ant_maze_medium_noisy_multistart_True_multigoal_False_sparse.hdf5',
        'non_zero_reset':False,
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-medium-diverse-render-v1',
    entry_point='trajectory.env.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.BIG_MAZE_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v1/Ant_maze_medium_noisy_multistart_True_multigoal_True_sparse.hdf5',
        'non_zero_reset':False,
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-large-diverse-render-v1',
    entry_point='trajectory.env.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.HARDEST_MAZE_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v1/Ant_maze_large_noisy_multistart_True_multigoal_True_sparse.hdf5',
        'non_zero_reset':False,
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-large-play-render-v1',
    entry_point='trajectory.env.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.HARDEST_MAZE_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v1/Ant_maze_large_noisy_multistart_True_multigoal_False_sparse.hdf5',
        'non_zero_reset':False,
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-eval-umaze-render-v0',
    entry_point='trajectory.env.locomotion.ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'maze_map': maze_env.U_MAZE_EVAL_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_umaze_eval_noisy_multistart_True_multigoal_False_sparse.hdf5',
        'non_zero_reset':False,
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-eval-umaze-diverse-render-v0',
    entry_point='trajectory.env.locomotion.ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'maze_map': maze_env.U_MAZE_EVAL_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_umaze_eval_noisy_multistart_True_multigoal_True_sparse.hdf5',
        'non_zero_reset':False,
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-eval-medium-play-render-v0',
    entry_point='trajectory.env.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.BIG_MAZE_EVAL_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_medium_eval_noisy_multistart_True_multigoal_True_sparse.hdf5',
        'non_zero_reset':False,
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-eval-medium-diverse-render-v0',
    entry_point='trajectory.env.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.BIG_MAZE_EVAL_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_medium_eval_noisy_multistart_True_multigoal_False_sparse.hdf5',
        'non_zero_reset':False,
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-eval-large-diverse-render-v0',
    entry_point='trajectory.env.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.HARDEST_MAZE_EVAL_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_large_eval_noisy_multistart_True_multigoal_False_sparse.hdf5',
        'non_zero_reset':False,
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-eval-large-play-render-v0',
    entry_point='trajectory.env.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.HARDEST_MAZE_EVAL_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_large_eval_noisy_multistart_True_multigoal_True_sparse.hdf5',
        'non_zero_reset':False,
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)
