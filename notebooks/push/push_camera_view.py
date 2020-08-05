from cmx import doc, md, csv
import gym
from uvpn.domains.sawyer import GoalImg

doc @ """
# Sawyer Push Task

# first step is to get PPO with HER to work
# then get TD3 with HER to work. Each should be a couple of hours.

## Top-down View
"""
with doc, doc.row():
    env = gym.make('sawyer:Push-v0', cam_id=0, mode="rgb", num_objs=1)
    env = GoalImg(env)
    obs = env.reset()
    doc.image(obs["img"].transpose([1, 2, 0]), "figures/push.png")
    doc.print(*obs.keys(), sep=",\n")

doc.flush()
