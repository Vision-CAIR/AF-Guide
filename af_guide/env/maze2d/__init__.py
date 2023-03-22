from d4rl.pointmaze.maze_model import U_MAZE, MEDIUM_MAZE, LARGE_MAZE
from gym.envs.registration import register



register(
    id='maze2d-umaze-sparse-v1',
    entry_point='trajectory.env.maze2d.maze2d:SparseMazeEnv',
    max_episode_steps=300,
    kwargs={
        'maze_spec':U_MAZE,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5'
    }
)

register(
    id='maze2d-medium-sparse-v1',
    entry_point='trajectory.env.maze2d.maze2d:SparseMazeEnv',
    max_episode_steps=600,
    kwargs={
        'maze_spec':MEDIUM_MAZE,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-medium-sparse-v1.hdf5'
    }
)


register(
    id='maze2d-large-sparse-v1',
    entry_point='trajectory.env.maze2d.maze2d:SparseMazeEnv',
    max_episode_steps=800,
    kwargs={
        'maze_spec':LARGE_MAZE,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-large-sparse-v1.hdf5'
    }
)