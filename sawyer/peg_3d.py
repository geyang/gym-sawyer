from collections import OrderedDict
import numpy as np
import gym
from gym.spaces import Box, Dict

from .env_util import get_asset_full_path
from .base import SawyerXYZEnv, SawyerCamEnv, pick


class SawyerPeg3DEnv(SawyerCamEnv, SawyerXYZEnv):

    def __init__(
            self,
            task=None,
            slot_low=None,
            slot_high=None,
            init_mode=None,
            goal_mode=None,
            cam_id=-1,
            **kwargs
    ):
        self.task = task

        model_name = get_asset_full_path(f'sawyer_peg_3d.xml')

        SawyerXYZEnv.__init__(self, model_name=model_name, **kwargs)
        SawyerCamEnv.__init__(self, cam_id=cam_id, **kwargs)

        self.init_mode = init_mode
        self.goal_mode = goal_mode

        # extend the observation space with objects
        self.slot_space = Box(np.array(slot_low), np.array(slot_high))
        d = pick(self.observation_space.spaces, *self.obs_keys)
        d[f'slot'] = self.slot_space
        self.observation_space = gym.spaces.Dict(**d)

    obs_keys = "hand", "slot"
    goal_keys = "hand",

    def get_slot_pos(self):
        return self.data.get_body_xpos('slot').copy()

    def _set_slot_xyz(self, pos, slot_id=0):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        offset = slot_id * 7
        qpos[9 + offset:12 + offset] = pos.copy()
        # qvel[9 + offset:12 + offset] = 0
        self.set_state(qpos, qvel)

    def _set_markers(self):
        self._set_slot_xyz(self.slot_pos)

    def state_dict(self):
        state = super().state_dict()
        return {**state, "slot": self.get_slot_pos()}

    # todo: might need to change the mode.
    def reset_model(self, mode=None, to_goal=None):
        """Provide high-level `mode` for sampling types of goal configurations."""
        self.sim.reset()
        mode = mode or (self.goal_mode if to_goal else self.init_mode)

        self.slot_pos = self.slot_space.sample()
        self._set_slot_xyz(self.slot_pos)

        rd = self.np_random.rand()
        if mode is None:
            if rd < 0.90:
                mode = 'hover'
            else:
                mode = "inserted"

        if mode == 'hover':  # hover

            self.gripper = self.np_random.choice([-1, 1], size=1)

            hand_pos = self.hand_space.sample()
            hand_pos += [0, 0, 0.03]
            self._reset_hand(hand_pos, [1, -1])

        elif mode == "inserted":

            self.gripper = 1  # always close the pincher

            hand_pos = self.slot_pos + [0, 0, 0.1]
            self._reset_hand(hand_pos, [1, -1])
            hand_pos = self.slot_pos + [0, 0, 0.03]
            self._reset_hand(hand_pos, [1, -1])
        else:
            raise NotImplementedError(f"{mode} is not supported")

        # todo: might be necessary
        # self._set_slot_xyz(self.slot_pos)
        return self.get_obs()


gym.envs.register(
    id="Peg3D-v0",
    entry_point=SawyerPeg3DEnv,
    # Place goal has to be on the surface.
    kwargs=dict(frame_skip=5,
                # init_mode="hover",
                goal_mode="inserted",
                mocap_low=(-0.1, 0.45, 0.05),
                mocap_high=(0.1, 0.55, 0.40),
                slot_low=(0.0, 0.5, 0.03),
                slot_high=(0.0, 0.5, 0.03)
                ),
    # max_episode_steps=100,
    reward_threshold=-3.75,
)
