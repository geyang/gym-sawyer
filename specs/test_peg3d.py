from cmx import doc
import gym
import numpy as np
from env_wrappers.flat_env import FlatGoalEnv
from sawyer.misc import space2dict, obs2dict
from tqdm import trange


def test_start():
    doc @ """
    # Sawyer Peg3D Environment
    We include the following domains in this test:
    """
    doc.csv @ """
    Name,        goal_keys,           Action Space,        Observation Space
    Peg3D-v0,    "hand"
    """


def test_peg3d():
    doc @ """
    ## sawyer:Peg3D-v0
    """
    with doc:
        env = gym.make("sawyer:Peg3D-v0")
    env.seed(100)
    obs = env.reset()
    with doc("Make sure that the spec agrees with what it returns"):
        doc.yaml(space2dict(env.observation_space))
    with doc:
        doc.yaml(obs2dict(obs))
    img = env.render('rgb')
    with doc.row() as row:
        row.image(img, caption="Peg3D-v0")
    doc.flush()


def test_peg3d_flat_goal_env():
    doc @ """
    To use with RL algorithms, use the FlatGoalEnv wrapper:
    """
    with doc:
        env = gym.make("sawyer:Peg3D-v0")
        env = FlatGoalEnv(env)
    env.seed(100)
    obs = env.reset()
    with doc("Which gives the following observation type"):
        doc.yaml(space2dict(env.observation_space))
    with doc:
        doc.yaml(obs2dict(obs))


def test_peg3d_reward():
    doc @ """
    ## sawyer:Reach-v0
    """
    with doc:
        env = gym.make("sawyer:Peg3D-v0")
    env.seed(100)

    frames = []
    obs = env.reset()
    for step in range(100):
        # gripper dimension does not matter
        act = obs['slot'] - obs['hand']
        obs, r, done, info = env.step(np.array([*act, -1]) * 10)
        img = env.render('rgb', width=240, height=240)
        frames.append(img)
        if done:
            break
    else:
        should_not_terminate = True
    assert should_not_terminate

    for ep in trange(4, desc="peg3d clipper closed"):
        obs = env.reset()
        for step in range(100):
            # gripper dimension does not matter
            delta = [0, 0, 0.2] if step < 20 else [0, 0, -0.04]
            act = obs['slot'] - obs['hand'] + delta
            obs, r, done, info = env.step(np.array([*act, 1]) * 10)
            img = env.render('rgb', width=240, height=240)
            frames.append(img)
            if done:
                break
        else:
            # print(ep, "is no good")
            raise RuntimeError("Reach failed to terminate")

    doc.video(frames, f"videos/peg3d_test.gif")
    doc.flush()
#
