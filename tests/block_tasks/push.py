from cmx import doc, md, csv
import gym
from tqdm import trange
from uvpn.domains.sawyer import GoalImg

doc @ """
# Sawyer Push Task

This is a simple semi-2D task. We restrict the configuration space
to keep this simple. In the future we can expand this to a larger
configuration space (3D as opposed to 2D).

There are three camera views:
"""

csv @ """
`cam_id`, Description
`-1`, human view 
`0`, gripper-centric view
`1`, Top-down view
"""
doc @ """
## Canonical View (`cam_id == -1`)
"""
with doc, doc.row():
    env = gym.make('sawyer:Push-v0', cam_id=-1, mode="rgb", num_objs=2)
    env = GoalImg(env)
    obs = env.reset()
    img = env.render('glamor', width=640, height=240)
    doc.image(img, "figures/push_glamor.png")


doc @ """
## All Views (`cam_id == 0`)

This view watched down from above.
"""
with doc, doc.row():
    for cam_id in [-1, 0, 1]:
        env = gym.make('sawyer:Push-v0', cam_id=cam_id, mode="rgb", num_objs=1, clipper=1)
        env = GoalImg(env)
        obs = env.reset()
        doc.image(obs["img"].transpose([1, 2, 0]), f"figures/push_{cam_id}.png", caption=f"cam_id={cam_id}")

doc @ """
## Distribution of blocks

This view watched down from above.
"""
with doc, doc.row():
    import numpy as np
    cam_id = 0
    env = gym.make('sawyer:Push-v0', cam_id=cam_id, mode="rgb", num_objs=1)
    env = GoalImg(env)
    images = []
    for i in trange(10):
        obs = env.reset()
        images.append(obs['img'])

    doc.image(np.array(images).min(0).transpose([1, 2, 0]), f"figures/push_rho_0.png")
    # doc.print(*obs.keys(), sep=",\n")
