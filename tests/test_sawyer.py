def test_pick_place():
    import gym

    env = gym.make("sawyer:PickPlace-v0", shaped_init=0.5, cam_id=0)
    env.reset()
    for _ in range(100):
        img = env.render("grey")
        assert img.max() > 0, "can not be all black"


if __name__ == '__main__':
    import numpy as np
    import gym

    # env = gym.make("sawyer:DoorReach-v0")
    env = gym.make("sawyer:PickPlace-v0", shaped_init=0.5, cam_id=0)
    # env = gym.make("sawyer:MixedMultitask-v0")
    for _ in range(10000):
        env.reset()
        for i in range(100):
            env.step(np.random.randn(4))
            env.render("human", width=640, height=640)
