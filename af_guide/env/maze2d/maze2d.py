from d4rl.pointmaze.maze_model import MazeEnv
from d4rl.locomotion import wrappers


class SparseMazeEnv(MazeEnv):
    def __init__(self, *args, **kargs):
        self.eval = False
        super().__init__(*args, **kargs)

    def step(self, action):
        ob, reward, done, info = super().step(action)
        if self.eval and reward > 0:
            done = True
        return ob, reward, done, info



