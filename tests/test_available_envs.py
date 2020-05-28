def test_available_envs():
    import gym
    gym.logger.set_level(40)  # turn off the warning

    envs = [
        'gym_sawyer:PointSingleTask-v0',
        'gym_sawyer:PointMultitaskSimple-v0',
        'gym_sawyer:PointMultitask-v0',
        'gym_sawyer:PickSingleTask-v0',
        'gym_sawyer:PickMultitaskSimple-v0',
        'gym_sawyer:PickMultitask-v0',
        'gym_sawyer:PickReachSingleTask-v0',
        'gym_sawyer:PickReachMultitaskSimple-v0',
        'gym_sawyer:PickReachMultitask-v0',
    ]

    for name in envs:
        env = gym.make(name)
        obs = env.reset()
        assert obs is not None, f"{name} is tested"
