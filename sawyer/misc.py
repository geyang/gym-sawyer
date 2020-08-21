from collections import Sequence

import gym
import numpy as np


def space2dict(space):
    if isinstance(space, gym.spaces.Dict):
        return {k: space2dict(v) for k, v in space.spaces.items()}
    elif isinstance(space, gym.spaces.Box):
        return f"shape{space.shape}"
    elif isinstance(space, gym.spaces.Discrete):
        return f"[{space.n}]"
    else:
        raise NotImplemented


def obs2dict(obs):
    if isinstance(obs, dict):
        return {k: obs2dict(v) for k, v in obs.items()}
    elif isinstance(obs, Sequence):
        return f"[{obs.__len__()}]"
    elif isinstance(obs, np.ndarray):
        return f"Shape{obs.shape}"
    else:
        return str(obs)
