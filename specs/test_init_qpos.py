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


def test_init_qpos():
    doc @ """
    ## sawyer:Peg3D-v0
    """
    with doc:
        env = gym.make("sawyer:Reach-v0")
        env.reset()
    img = env.render('rgb')
    with doc.row() as row:
        row.image(img, caption="Reach-v0")
    doc.flush()



