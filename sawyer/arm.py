import numpy as np
from gym import Wrapper
from gym.spaces import Box, Dict

from .env_util import get_asset_full_path
from .base import SawyerXYZEnv, SawyerCamEnv

# ALL_TASKS = [
#     "pick",
#     "pick_place",
#     "stack"
# ]

geom_types = {None: 5, 'cylinder': 5, 'box': 6}
geom_xy = {None: 1, 'cylinder': 1, 'box': 2}


class SawyerArmEnv(SawyerCamEnv, SawyerXYZEnv):

    def __init__(
            self,
            task=None,
            fixed_goal=None,
            cam_id=-1,
            **kwargs
    ):
        self.task = task

        model_name = get_asset_full_path(f'sawyer_arm.xml')

        SawyerXYZEnv.__init__(self, model_name=model_name, **kwargs)
        SawyerCamEnv.__init__(self, cam_id=cam_id, **kwargs)

        self.fixed_goal = None if fixed_goal is None else np.array(fixed_goal)

    obs_keys = "hand",
    goal_keys = "hand",


from gym.envs import register

register(
    id="Reach-v0",
    entry_point=SawyerArmEnv,
    kwargs=dict(frame_skip=5,
                gripper=1,
                mocap_low=(-0.25, 0.3, 0.06),
                mocap_high=(0.25, 0.7, 0.30),
                ),
    # max_episode_steps=100,
    # reward_threshold=-3.75,
)
