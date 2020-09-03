from cmx import doc
import gym
import numpy as np
from env_wrappers.flat_env import FlatGoalEnv
from sawyer.misc import space2dict, obs2dict

ALL_ENVS = [
    "sawyer:PickPlace-v0",
    "sawyer:Peg3D-v0",
]


def test_start():
    doc @ """
    # Sawyer Manipulation Task Suite
    
    We want: single-task environments on which we can verify the HER
    implementation and ρ₀, 
    
    ## Environments Needs to be Built
    
    - [ ] `Box-open-v0`: make the box transparent
    - [ ] `Box-close-v0`
    - [ ] `Bin-picking-v0`: two variants, to right, to left.
    - [ ] `Bin-picking-v0`
    - [ ] `Drawer-open-v0`
    - [ ] `Drawer-close-v0`
    
    We include the following domains in this test:
    """
    doc.csv @ """
    Name,        goal_keys,           Action Space,        Observation Space
    Reach-v0,    "hand"
    Push-v0,     "obj_0"
    PushMove-v0, "hand"&#44; "obj_0"
    """
    doc.flush()
