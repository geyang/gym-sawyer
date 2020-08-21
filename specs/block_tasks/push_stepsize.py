from cmx import doc, md, csv
import gym
from tqdm import tqdm
from uvpn.domains.sawyer import GoalImg

doc @ """
# Sawyer Push Task (Step Size)

How big of a step can the arm take?

- assume continuous control
- normalize the action space: what is the action norm?
- restrict the action dimension to 2D

With an action scale of 2/100, it takes about 16 time steps
to get to the other side.

"""
import numpy as np

with doc, doc.row():
    task = {"hand_pos": [-0.25, 0.25], "obj_pos": [-0.15, 0.35]}, {"hand_pos": [0.25, 0.75], "obj_pos": [0.15, 0.65]}

    def normalize(act):
        norm = np.linalg.norm(act)
        return act / norm

    act = np.array(task[1]['hand_pos']) - np.array(task[0]['hand_pos'])

    env = gym.make('sawyer:Push-v0', cam_id=0, width=240, height=240)
    env = GoalImg(env)
    env.seed(100)
    env.reset(x_kwargs=task[0], goal_kwargs=task[1])
    all_images = []
    for step in range(20):
        obs, reward, done, info = env.step(act)
        all_images.append(obs['img'].transpose([1, 2, 0]))
        doc.image(np.array(all_images).min(axis=0), caption=f"Step {step}")

    doc.image(np.array(all_images).min(axis=0), f"figures/stepsize/overlay.png", caption=f"")
