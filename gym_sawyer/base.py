import abc
import numpy as np
import mujoco_py
from .mujoco_env import MujocoEnv
import copy


class SawyerMocapBase(MujocoEnv, metaclass=abc.ABCMeta):
    """
    Provides some commonly-shared functions for Sawyer Mujoco envs that use
    mocap for XYZ control.
    """
    mocap_low = np.array([-0.2, 0.5, 0.05])
    mocap_high = np.array([0.2, 0.7, 0.6])

    def __init__(self, model_name, frame_skip=None, **kwargs):
        MujocoEnv.__init__(self, model_name, frame_skip=5 if frame_skip is None else frame_skip, **kwargs)
        # Resets the mocap welds that we use for actuation.
        sim = self.sim
        if sim.model.nmocap > 0 and sim.model.eq_data is not None:
            for i in range(sim.model.eq_data.shape[0]):
                if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    # Define the xyz + quat of the mocap relative to the hand
                    sim.model.eq_data[i, :] = np.array([0., 0., 0., 1., 0., 0., 0.])

    def reset_mocap2body_xpos(self):
        # move mocap to weld joint
        pos = self.data.get_body_xpos('hand').copy()
        quat = self.data.get_body_quat('hand').copy()
        self.data.set_mocap_pos('mocap', np.array([pos]), )
        self.data.set_mocap_quat('mocap', np.array([quat]), )

    def get_mocap_pos(self):
        return self.data.get_site_xpos('mocap').copy()

    def get_endeff_pos(self):
        return self.data.get_body_xpos('hand').copy()

    def get_endeff_vel(self):
        grip_velp = self.data.get_body_xvelp('hand').copy()
        return grip_velp

    def get_gripper_pos(self):
        left = self.data.get_site_xpos('leftEndEffector').copy()
        right = self.data.get_site_xpos('rightEndEffector').copy()
        return left, right

    def get_env_state(self):
        joint_state = self.sim.get_state()
        mocap_state = self.data.mocap_pos, self.data.mocap_quat
        state = joint_state, mocap_state
        return copy.deepcopy(state)

    def set_env_state(self, state):
        joint_state, mocap_state = state
        self.sim.set_state(joint_state)
        mocap_pos, mocap_quat = mocap_state
        self.data.set_mocap_pos('mocap', mocap_pos)
        self.data.set_mocap_quat('mocap', mocap_quat)
        self.sim.forward()


class SawyerXYZEnv(SawyerMocapBase, metaclass=abc.ABCMeta):
    def __init__(
            self,
            *args,
            mocap_low=(-0.1, 0.5, 0.05),
            mocap_high=(0.1, 0.7, 0.6),
            effector_quat=(1, 0, 1, 0),
            action_scale=1 / 100,
            show_mocap=True,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.effector_quat = np.array(effector_quat)
        self.action_scale = action_scale
        # note: separate mocap range vs object range.
        self.mocap_low = np.hstack(mocap_low)
        self.mocap_high = np.hstack(mocap_high)

        if not show_mocap:
            self.hide_mocap()

    def hide_mocap(self):
        self.sim.model.geom_rgba[3:21] = 0
        self.sim.model.geom_rgba[33:37] = 0

    def set_xyz_action(self, action):
        action = np.clip(action, -1, 1)
        pos_delta = action * self.action_scale
        new_mocap_pos = self.data.mocap_pos + pos_delta
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high,
        )
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        self.data.set_mocap_quat('mocap', self.effector_quat)

    def _reset_hand(self, pos=None):
        if pos is None:
            pos = self.hand_init_pos
        for _ in range(10):
            self.data.set_mocap_pos('mocap', pos)
            self.data.set_mocap_quat('mocap', self.effector_quat)
            self.do_simulation(None, self.frame_skip)


class SawyerCamEnv(metaclass=abc.ABCMeta):
    def __init__(self, *args, cam_id=None, width=100, height=100, **kwargs):
        self.cam_id = cam_id
        self.width = width
        self.height = height

    def viewer_setup(self, cam_id=None):
        cam_id = self.cam_id if cam_id is None else cam_id
        camera = self.viewer.cam
        if cam_id == -1:
            camera.fixedcamid = cam_id
            camera.type = mujoco_py.generated.const.CAMERA_FREE
        elif cam_id is not None:
            camera.fixedcamid = cam_id
            camera.type = mujoco_py.generated.const.CAMERA_FIXED
