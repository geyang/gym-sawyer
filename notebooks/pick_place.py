from cmx import doc, md, csv
import gym

md @ """
# Sawyer Pick and Place Environment
"""

with md @ """
## Number of Objects
can include 1 - 4 objects:""", doc.row():
    for i in range(1, 5):
        env = gym.make("sawyer:PickPlace-v0", cam_id=-1, num_objs=i)
        env.reset()
        img = env.render("glamor", width=150, height=150)
        doc.image(img, src=f"./figures/pick_place_{i}.png", caption=f"{i} block{'s' if i > 1 else ''}")

md @ """
## Whitebox evaluation of the learned representation:

How well does the learned representation $\phi$ 
predict information such as object location and pose?

Here is our experiment grid:
"""
csv @ """
Method,             block,  arm,    Comments
naïve,              None,   None,   Only the robotic arm is varied a lot&#44; so there is not much variation to the location of the blocks.

"""

if __name__ == '__main__':
    doc.flush()
