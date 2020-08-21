from cmx import doc, md, csv
import gym
import numpy as np
from termcolor import cprint
from tqdm import trange
from ml_logger import logger
from uvpn.domains.sawyer import GoalImg

doc @ """
# Sawyer Push Task (Reward)

Relabeling using the sparse reward does not help, because
much of the MDP is disentangled from the robot unless the
gripper is touching the object.

## Standard push environment
"""
with doc, doc.row():
    env = gym.make('sawyer:Push-v0', cam_id=0)
    env = GoalImg(env)
    env.seed(100)
    obs = env.reset()
    for step in range(100):
        delta = obs['goal'] - obs['x']
        act = delta[:4] * [5, 5, 0, 1]
        obs, reward, done, info = env.step(act)
        doc.image(obs['img'].transpose([1, 2, 0]), f"figures/sequence/push_step_{step}.png", caption=f"Step {step}")
        if np.linalg.norm(obs['x'][:3] - obs['goal'][:3]) < 0.01:
            break
    doc.image(obs['goal_img'].transpose([1, 2, 0]), "figures/sequence/push_goal.png", caption="Goal Image")

with doc:
    for key, value in obs.items():
        doc.print(key, value.shape)
