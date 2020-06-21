import gym
import numpy as np


# wrapper classes are anti-patterns.
def FlatGoalEnv(env, obs_keys, goal_keys):
    """
    We require the keys to be passed in explicitly, to avoid mistakes.

    :param env:
    :param obs_keys: obs_keys=('state_observation',)
    :param goal_keys: goal_keys=('desired_goal',)
    """
    goal_keys = goal_keys or []

    for k in obs_keys:
        assert k in env.observation_space.spaces

    for k in goal_keys:
        assert k in env.observation_space.spaces
        
    assert isinstance(env.observation_space, gym.spaces.Dict)

    _observation_space = env.observation_space
    _step = env.step
    _reset = env.reset

    # TODO: handle nested dict
    env._observation_space = _observation_space
    env.observation_space = gym.spaces.Box(
        np.hstack([_observation_space.spaces[k].low for k in obs_keys]),
        np.hstack([_observation_space.spaces[k].high for k in obs_keys]),
    )

    if len(goal_keys) > 0:
        env.goal_space = gym.spaces.Box(
            np.hstack([_observation_space.spaces[k].low for k in goal_keys]),
            np.hstack([_observation_space.spaces[k].high for k in goal_keys]),
        )

    # _goal = None

    def step(action):
        nonlocal obs_keys
        obs, reward, done, info = _step(action)
        flat_obs = np.hstack([obs[k] for k in obs_keys])
        return flat_obs, reward, done, info

    def reset():
        nonlocal goal_keys
        obs = _reset()
        # if len(goal_keys) > 0:
        #     _goal = np.hstack([obs[k] for k in goal_keys])
        return np.hstack([obs[k] for k in obs_keys])

    # def get_goal(self):
    #     return _goal

    env.step = step
    env.reset = reset

    return env
