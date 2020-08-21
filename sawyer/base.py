import abc
import numpy as np
import mujoco_py
from termcolor import cprint
from gym import spaces

from .mujoco_env import MujocoEnv
import copy


def pick(d: dict, *keys):
    if not keys:  # info: return original not a copy
        return d
    return {k: d[k] for k in keys if k in d}


class SawyerMocapBase(MujocoEnv, metaclass=abc.ABCMeta):
    """
    Provides some commonly-shared functions for Sawyer Mujoco envs that use
    mocap for XYZ control.
    """

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
        cprint('reset_mocap2body_xpos', "green")
        # move mocap to weld joint
        pos = self.data.get_body_xpos('hand').copy()
        quat = self.data.get_body_quat('hand').copy()
        self.data.set_mocap_pos('mocap', np.array([pos]), )
        self.data.set_mocap_quat('mocap', np.array([quat]), )

    def get_mocap_pos(self):
        cprint('get_mocap_xpos', "green")
        return self.data.get_site_xpos('mocap').copy()

    def get_endeff_pos(self):
        return self.data.get_body_xpos('hand').copy()

    def get_endeff_vel(self):
        cprint('get_endeff_vel', "green")
        grip_velp = self.data.get_body_xvelp('hand').copy()
        return grip_velp

    def get_gripper_poses(self):
        left = self.data.get_site_xpos('leftEndEffector').copy()
        right = self.data.get_site_xpos('rightEndEffector').copy()
        return left, right

    def get_gripper_pos(self):
        """returns COM of gripper"""
        # cprint('get_gripper_pos', "green")
        l, r = self.get_gripper_poses()
        return (l + r) / 2

    def get_gripper_state(self):
        l, r = self.get_gripper_poses()
        d = np.linalg.norm(r - l)
        # info: [0.2 - 1] ± tolerance of the simulator
        return np.array([d / 0.10])
        # """returns 1 if gripper is closed, -1 otherwise"""
        # return np.array([[-1, 1][np.linalg.norm(l - r) < 0.07]])

    def state_dict(self):
        """child class can extend this"""
        return dict(
            hand=self.get_endeff_pos(),
            gripper=self.get_gripper_state(),
        )

    def _set_markers(self):
        """Set position for marker objects, called at the beginning of get_obs"""
        pass

    # def get_env_state(self):
    #     joint_state = self.sim.get_state()
    #     mocap_state = self.data.mocap_pos, self.data.mocap_quat
    #     state = joint_state, mocap_state
    #     return copy.deepcopy(state)
    #
    # def set_env_state(self, state):
    #     joint_state, mocap_state = state
    #     self.sim.set_state(joint_state)
    #     mocap_pos, mocap_quat = mocap_state
    #     self.data.set_mocap_pos('mocap', mocap_pos)
    #     self.data.set_mocap_quat('mocap', mocap_quat)
    #     self.sim.forward()


class SawyerXYZEnv(SawyerMocapBase, metaclass=abc.ABCMeta):
    def __init__(
            self,
            *args,
            mocap_low=(-0.1, 0.5, 0.05),
            mocap_high=(0.1, 0.7, 0.6),
            # this is the attitude of the end effector
            gripper=None,  # initial gripper state
            effector_quat=(1, 0, 1, 0),
            action_scale=2 / 100,
            show_mocap=True,
            r=0.02,  # this is the termination threshold
            obs_keys=None,
            goal_keys=None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.gripper_init = gripper

        self.effector_quat = np.array(effector_quat)
        self.action_scale = action_scale
        # note: separate mocap range vs object range.
        self.mocap_low = np.hstack(mocap_low)
        self.mocap_high = np.hstack(mocap_high)
        self.r = r

        self.goal_keys = goal_keys or self.goal_keys
        self.action_space = spaces.Box(low=np.full(4, -1), high=np.ones(4))
        # used to sample hand pos
        self.hand_space = spaces.Box(low=self.mocap_low, high=self.mocap_high)
        d = dict(hand=self.hand_space,
                 gripper=spaces.Box(low=np.array([0]), high=np.array([1])))

        if obs_keys:
            # avoid raising KeyError when child class has different obs_keys.
            self.obs_keys = obs_keys
            d = pick(d, *self.obs_keys)

        self.observation_space = spaces.Dict(**d)
        if not show_mocap:
            self.hide_mocap()

    # observation does not contain the goal.
    obs_keys = "hand", "gripper"
    goal_keys = None

    def set_goal(self, obs):
        """override this method in the child class"""
        keys = self.goal_keys or obs.keys()
        self.goal = pick(obs, *keys)

    def get_obs(self):
        self._set_markers()
        return pick(self.state_dict(), *self.obs_keys)

    def compute_reward(self, x, goal, info=False):
        """this returns the distance to goal, in a way specific
        to the robotics environment. Can be overridden by the
        child class."""
        d = 0
        for k in goal:
            d += np.linalg.norm(goal[k] - x[k])
        d /= len(goal)  # normalize the distance

        done = d < self.r
        r = float(done) - 1
        if info:
            return r, done, dict(dist=d, success=d < self.r)
        return r

    def compute_rewards(self, xs, goals):
        """
        Need to be consistent with the singular counterpart.

        Note: It looks like for `push-move`, even with re-labeling
            the number of positive rewards is going to be very few,
        """
        d = 0
        for k in goals:
            d += np.linalg.norm(goals[k] - xs[k], axis=-1)
        d /= len(goals)  # normalize the distance
        done = d < self.r
        r = float(done) - 1
        return r

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

    def _reset_hand(self, pos=None, gripper=None, steps=30):
        if pos is None:
            pos = self.hand_init_pos
        self.data.set_mocap_pos('mocap', pos)
        self.data.set_mocap_quat('mocap', self.effector_quat)
        self.do_simulation(gripper, n_frames=steps)

    def reset_model(self, hand_pos=None, to_goal=None):
        """Provide high-level `mode` for sampling types of goal configurations."""
        self.sim.reset()

        # always drop object 0 from the air.
        if hand_pos is not None:
            if self.gripper_init is not None:
                self.gripper = self.gripper_init
            else:
                self.gripper = self.np_random.choice([-1, 1], size=1)
            self._reset_hand(hand_pos, [self.gripper, -self.gripper])
            return self.get_obs()

        hand_pos = self.hand_space.sample()
        # note: this is the only free wavering mode
        if self.gripper_init is not None:
            self.gripper = self.gripper_init
        else:
            self.gripper = self.np_random.choice([-1, 1], size=1)
        self._reset_hand(hand_pos, [self.gripper, -self.gripper])

        return self.get_obs()

    def reset(self):
        """We call self.reset_model for a single reset (it is currently expensive)"""
        obs = self.reset_model(to_goal=True)
        self.set_goal(obs)
        obs = self.reset_model()
        return obs

    def step(self, action):
        """termination requires access to reward, so self.dt also requires reward."""
        self.set_xyz_action(action[:3])

        gripper = action[3]

        if gripper > 0:
            self.gripper = 1
        elif gripper < 0:
            self.gripper = -1

        for i in range(self.frame_skip):
            self.do_simulation((self.gripper, -self.gripper), n_frames=1)
            ob = self.get_obs()
            reward, done, info = self.compute_reward(ob, self.goal, info=True)
            if done:
                break

        return ob, reward, done, info


class SawyerCamEnv(metaclass=abc.ABCMeta):
    def __init__(self, *args, cam_id=None, width=100, height=100, **kwargs):
        self.cam_id = cam_id
        self.width = width
        self.height = height

    def viewer_setup(self):
        """The camera id here is not real."""
        cam_id = self.cam_id
        camera = self.viewer.cam

        if cam_id == -1:
            # ortho-view
            camera.azimuth = -135
            camera.elevation = -25
            camera.trackbodyid = -2
            camera.lookat[0] = 0
            camera.lookat[1] = .5
            camera.lookat[2] = 0.2
            camera.distance = 1.6

# class FlatGoal(Wrapper):
#
#     def __init__(self, env, obs_keys=None):
#         super().__init__(env)
#         obs_keys = obs_keys or env.obs_keys
#         # self.observation_space = {k: v for k, v in env.observation_space.spaces.items()}
#         obspc = self.observation_space.spaces
#         self.observation_space = spaces.Box(*zip(*[(obspc[k].low, obspc[k].hight) for k in obs_keys]))
#
#     def reset(self):
#         obs = self.reset_model()
#         self.goal = obs['x'].copy()
#         obs = self.reset_model()
#         return obs
#
#     def get_obs(self):
#         self._set_markers()
#         obs = self.state_dict()
#         obs['x'] = np.concatenate([obs[k] for k in self.FLAT_KEYS])
#         return obs
