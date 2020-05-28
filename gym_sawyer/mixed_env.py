"""
The sawyer robot starts in the center, because relying on starting from the
left is unreliable. We now shape the reward to encourage the agent to lift
the arm.

Entropy bonus does not work super well with these Sawyer environments. Exploration should
be done via other means, or provided by demonstrations.

                                                                   -- Ge
"""
from collections import OrderedDict
from functools import reduce
from operator import iadd

import numpy as np
from gym.spaces import Box, Dict
import mujoco_py.generated

from .env_util import get_stat_in_paths, create_stats_ordered_dict, get_asset_full_path
from .multitask_env import MultitaskEnv
from .base import SawyerXYZEnv, SawyerCamEnv


class Controls:
    def __init__(self, k_tasks, seed=None):
        """
        The task index is always initialized to be 0. Which means that unless you sample task, this
        is the same as the single-task baseline. So if you want to use this in a multi-task setting,
        make sure that you resample obj_goal each time after resetting the environment.

        :param k_tasks:
        :param seed:
        """
        self.k = k_tasks
        # deterministically generate the obj_goal so that we don't have to pass in the positions.
        self.rng = np.random.RandomState(seed)
        self.index = 0  # this is always initialized to be 0.

    def seed(self, seed):
        self.rng.seed(seed)

    def sample_task(self, index=None):
        if index is None:
            self.index = self.rng.randint(0, self.k)
        else:
            self.index = index
            assert index < self.k, f"index need to be less than the number of tasks {self.k}."
        return self.index


class SawyerMixedMultitaskEnv(MultitaskEnv, SawyerXYZEnv):
    def __init__(
            self,
            k_tasks=6,
            sample_task_on_reset=False,
            obj_init_pos=None,
            hand_init_pos=None,
            obj_low=(-0.35, 0.3, 0.02),
            obj_high=(0.35, 0.6, 0.3),
            fixed_goal=None,
            reward_type="door_dense",
            obj_in_hand=0,
            cam_id=-1,
            width=84, height=84,
            **kwargs
    ):
        self.controls = Controls(k_tasks=k_tasks)
        MultitaskEnv.__init__(self)
        SawyerXYZEnv.__init__(self, model_name=self.model_name, **kwargs)
        SawyerCamEnv.__init__(self, cam_id=cam_id, widht=width, height=height)

        self.obj_in_hand = obj_in_hand

        self.fixed_goal = fixed_goal
        self.sample_task_on_reset = sample_task_on_reset
        self.reward_type = reward_type
        self.obj_init_pos = None if obj_init_pos is None else np.array(obj_init_pos)
        self.hand_init_pos = None if hand_init_pos is None else np.array(hand_init_pos)
        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
        )
        self.obj_space = Box(
            np.concatenate([obj_low] * self.k_obj),
            np.concatenate([obj_high] * self.k_obj),
        )
        self.obj_goal_space = Box(np.array(obj_low), np.array(obj_high), )
        self.hand_space = Box(self.mocap_low, self.mocap_high)
        self.door_space = Box(np.array([0, 0] * self.k_doors), np.array([1, 1] * self.k_doors))
        self.hand_door_delta_space = Box(np.array([-1] * self.k_doors * 3), np.array([1] * self.k_doors * 3))
        self.gripper_space = Box(np.array([-0.03] * 4), np.array([0.03] * 4), dtype=float)
        # self.task_space = Discrete(k_doors)
        self.task_space = Box(np.zeros(k_tasks), np.ones(k_tasks), dtype=float)
        """
        The gym reacher environment returns the following.
            cos(theta), sin(theta), goal, theta_dot, delta
        Here instead, we return
            cos(theta), sin(theta), theta_dot, goal_posts (target + distractor), delta (with fingertip)
        We don't ever return the position of the finger tip.
        """
        self.observation_space = Dict(dict(
            hand_pos=self.hand_space,
            hand_dot=self.hand_space,
            obj_poses=self.obj_space,
            door_poses=self.door_space,  # the x, y position of the door sites
            hand_obj_delta=self.obj_space,
            hand_door_delta=self.hand_door_delta_space,
            # note: we don't have a goal for objects yet.
            obj_goal=self.obj_goal_space,
            obj_goal_delta=self.obj_space,
            gripper=self.gripper_space,
            gripper_pos=self.hand_space,
            task_id=self.task_space,
        ))

    @property
    def model_name(self):
        assert self.controls.k == 6, f'Need to have 6 tasks, currently has {self.controls.k}'
        return get_asset_full_path(f'sawyer_mixed_multitask.xml')

    def viewer_setup(self, cam_id=None):
        SawyerCamEnv.viewer_setup(self, self.cam_id if cam_id is None else cam_id)

        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = .65
        self.viewer.cam.lookat[2] = 0.2
        self.viewer.cam.distance = 1.6
        self.viewer.cam.elevation = -15
        self.viewer.cam.azimuth = 135

    def step(self, action):
        self.set_xyz_action(action[:3])
        self.do_simulation((action[3], -action[3]))
        # The marker seems to get reset every time you do a simulation
        ob = self._get_obs()
        reward = self.compute_reward(action, ob)
        info = self._get_info(ob)
        done = False
        return ob, reward, done, info

    k_doors = 3
    k_obj = 3
    task_ids = np.eye(6, 6, )

    def _get_obs(self):
        e = self.get_endeff_pos()
        doors = self.get_door_poses()
        dot = self.get_endeff_vel()
        d = self.get_obj_distractor_pos()  # note: we keep the ordering the same, just use different rewards
        l, r = self.get_gripper_pos()
        obj_goal = self._state_goal

        return dict(
            hand_pos=e,
            hand_dot=dot,
            obj_poses=np.concatenate(d),
            door_poses=np.concatenate([[x, y] for x, y, z in doors]),
            hand_obj_delta=np.concatenate(d - e),
            hand_door_delta=np.concatenate(doors - e),
            obj_goal=obj_goal,
            obj_goal_delta=np.concatenate(d - obj_goal),
            gripper=np.concatenate([l[:2] - e[:2], r[:2] - e[:2]]),
            gripper_pos=np.mean([l, r], axis=0),
            task_id=self.task_ids[self.controls.index],
        )

    def _get_info(self, ob):
        if self.reward_type == "dense":
            index = self.controls.index
            if index < self.k_doors:
                # door success
                door_poses = ob['door_poses']
                y_targ = door_poses[2 * self.controls.index + 1] - 0.76
                is_successful = y_targ < - 0.09
            else:
                # obj reach success
                index -= self.k_doors
                obj_pos = ob['obj_poses'][3 * index:3 + 3 * index]
                obj_goals = ob['obj_goal']
                obj_dist = np.linalg.norm(obj_goals - obj_pos, axis=-1)
                is_successful = obj_dist < 0.02
            return dict(success=is_successful)
        else:  # do NOT return anything in "point_dense" mode
            return {}

    def get_obj_pos(self):
        return self.data.get_body_xpos('obj').copy()

    def get_distractor_pos(self):
        return [self.data.get_body_xpos(f'distractor_{i}').copy() for i in range(1, self.k_obj)]

    def get_obj_distractor_pos(self):
        return [self.get_obj_pos(), *self.get_distractor_pos()]

    def get_door_poses(self):
        return [self.data.get_site_xpos(f"door_{i}_handle") for i in range(self.k_doors)]

    def _set_goal_marker(self):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        self.data.site_xpos[self.model.site_name2id('obj-goal-site')] = self._state_goal

    _before_render = _set_goal_marker

    def _set_hand_goal(self, goal):
        """debugging function"""
        self.data.site_xpos[self.model.site_name2id('hand-goal-site')] = goal

    def _set_obj_goal(self, goal):
        """debugging function"""
        self.data.site_xpos[self.model.site_name2id('obj-goal-site')] = goal

    # def _set_obj_xyz(self, pos):  # 3 * k objects
    #     assert len(pos) == 3, 'need to be 3D'
    #     qpos = self.data.qpos.flat.copy()
    #     qvel = self.data.qvel.flat.copy()
    #     qpos[9:12] = pos
    #     qvel[9:15] = 0
    #     self.set_state(qpos, qvel)

    # def _set_obj_distractor_xyz(self, pos, offset=9 + k_doors * 2, vel_offset=9 * 2):
    #     # pos = x, y, z + quarterion
    #     # vel = {x, y, z, theta, phi, gamma}_dot
    #     qpos = self.data.qpos.flat.copy()
    #     qvel = self.data.qvel.flat.copy()
    #     for i in range(self.k_doors, self.k_doors + self.k_obj):
    #         qpos[offset + i * 7:offset + i * 7 + 3] = pos[i * 3: i * 3 + 3]
    #     qvel[vel_offset:vel_offset + len(pos) / 3 * 6] = 0
    #     self.set_state(qpos, qvel)

    def sample_objects(self, floored=True):
        pos = np.random.uniform(self.obj_space.low, self.obj_space.high)
        if floored:
            pos[2::3] = 0.02
        return pos

    def sample_goals(self, batch_size, p_obj_in_hand=0.5):
        assert self.controls.k >= self.k_doors, \
            f"need to be a block task, but got {self.controls.k}"
        if self.fixed_goal is not None:
            goals = np.repeat(self.fixed_goal.copy()[None], batch_size, 0)
        else:
            goals = np.random.uniform(
                self.obj_goal_space.low,
                self.obj_goal_space.high,
                size=(batch_size, self.obj_goal_space.low.size),
            )
        # note: this is required because we call self.sample_goal, which wraps around self.sample_goals.
        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }
        # return dict(door_index=[self.controls.index])

    def set_task(self, id):
        self.controls.index = id

    def reset_model(self):
        if self.sample_task_on_reset:
            self.controls.sample_task()

        if self.obj_in_hand:
            put_in = np.random.rand(1) <= self.obj_in_hand
            if put_in:
                self.put_obj_in_hand(0.03)

        self._state_goal = self.sample_goal()['state_desired_goal']

        self._reset_hand()
        obj_pos = self.sample_objects() if self.obj_init_pos is None else self.obj_init_pos
        self._reset_doors_objs(obj_pos=obj_pos)
        return self._get_obs()

    def _reset_doors_objs(self, obj_pos=None):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        # set the doors to close
        _ = 9 + self.k_doors
        qpos[9:_] = 0
        qvel[9:_] = 0
        # set the objects
        for i in range(self.k_obj):
            qpos[_ + i * 7:_ + i * 7 + 3] = obj_pos[i * 3: i * 3 + 3]
        qvel[_:_ + len(obj_pos) * 3] = 0
        self.set_state(qpos, qvel)
        self.do_simulation([0, 0], 10)

    def _reset_hand(self, pos=None):
        if pos is None:
            pos = np.random.uniform(self.mocap_low,
                                    self.mocap_high) if self.hand_init_pos is None else self.hand_init_pos
        self.data.set_mocap_pos('mocap', pos)
        self.data.set_mocap_quat('mocap', self.effector_quat)
        self.do_simulation([0, 0], 10)

    def put_obj_in_hand(self, std: float = 0):
        noise = np.random.normal(scale=std, size=3) if std else 0
        for i in range(4):
            new_obj_pos = self.data.get_site_xpos('endEffector').copy()
            self._set_obj_xyz(new_obj_pos + noise)
            self.do_simulation([1, -1], 10)

    def set_to_goal(self, goal):
        state_goal = goal['state_desired_goal']
        self.data.set_mocap_pos('mocap', self.hand_init_pos)
        self.data.set_mocap_quat('mocap', self.effector_quat)
        # keep gripper closed
        self.do_simulation([1, -1], 10)
        self._set_obj_xyz(state_goal)
        self.sim.forward()

    """
    Multitask functions
    """

    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }

    def compute_rewards(self, actions, obs):
        if self.reward_type == "dense":
            if self.controls.index < self.k_doors:
                return self.door_rewards("open_dense", actions, obs, self.controls.index)
            else:
                return self.obj_rewards("pick_reach_dense", actions, obs, self.controls.index - self.k_doors)
        elif self.reward_type == "point_dense":
            if self.controls.index < self.k_doors:
                return self.door_rewards("reach_distance", actions, obs, self.controls.index)
            else:
                return self.obj_rewards("hover_dense", actions, obs, self.controls.index - self.k_doors)
        else:
            raise NotImplementedError(f"{self.reward_type} is not implemented")

    @staticmethod
    def obj_rewards(reward_type, actions, obs, index):
        gripper_state = obs['gripper']
        gripper_pos = obs['gripper_pos']
        hand_pos = obs['hand_pos']
        obj_pos = obs['obj_poses'][:, 3 * index:3 + 3 * index]
        obj_goals = obs['obj_goal']

        # hand_distances = np.linalg.norm(obj_goals - hand_pos, axis=1)
        # obj_distances = np.linalg.norm(obj_goals - obj_pos, axis=1)
        # hand_and_obj_distances = hand_distances + obj_distances
        touch_distances = np.linalg.norm(hand_pos - obj_pos, axis=1)
        # touch_and_obj_distance = touch_distances + obj_distances
        mocap_error = np.linalg.norm(hand_pos - gripper_pos - [0, 0, 0.035], axis=-1)

        threshold = 0.03

        if reward_type == 'touch_distance':
            r = -touch_distances
        elif reward_type == 'touch_success':
            r = -(touch_distances < threshold).astype(float)
        elif reward_type == 'hover_dense':
            _ = obj_pos - hand_pos + [0, 0, 0.035] + [0, 0, 0.05]
            _ = np.linalg.norm(_, axis=1)
            r = -_
        elif reward_type == 'touch_dense':
            offset = np.zeros_like(hand_pos)
            offset[:, -1] = - 0.03
            offset[:, :2] = (gripper_state[:, :2] + gripper_state[:, 2:]) / 2
            _ = obj_pos - (hand_pos + offset)
            touch_distances = np.linalg.norm(_, axis=1)
            touch_distances_xy = np.linalg.norm(_[:, :2], axis=1)
            gripper_opening = np.linalg.norm(gripper_state[:, :2] - gripper_state[:, 2:])
            dh = 0.1 - hand_pos[:, -1]
            h = hand_pos[:, -1] - 0.04 - 0.03
            opening = (gripper_opening - 0.04) / 2 + h * 1
            r = -touch_distances - (0 if touch_distances_xy < opening else 10) * np.maximum(dh, 0)
        elif reward_type == 'pick_sparse':
            # fixit: this is inconsistent with the dense reward below. Need to update
            r = (obj_pos[:, -1] > 0.025).astype(float)
        elif reward_type == 'pick_dense':
            shape_fn = lambda x: np.select([x > -0.3], [10 * (x + 0.3) ** 2 + x], x)
            offset = np.zeros_like(hand_pos)
            offset[:, -1] = - 0.03
            offset[:, :2] = (gripper_state[:, :2] + gripper_state[:, 2:]) / 2
            _ = obj_pos - (hand_pos + offset)
            touch_distances = np.linalg.norm(_, axis=1)
            touch_distances_xy = np.linalg.norm(_[:, :2], axis=1)
            gripper_opening = np.linalg.norm(gripper_state[:, :2] - gripper_state[:, 2:])
            dh = 0.083 - hand_pos[:, -1]
            opening = 0.02
            height_penalty = (-10 if touch_distances_xy > opening else 0) * np.maximum(dh, 0)
            grip_reward = (5 * (0.1 - gripper_opening)) if touch_distances_xy < opening and (
                    hand_pos[:, -1] - obj_pos[:, -1]) < 0.065 else ((gripper_opening - 0.1) * 5)
            # add all rewards
            r = shape_fn(-touch_distances)
            r += height_penalty
            r += grip_reward
            r -= mocap_error * 10
        elif reward_type == 'pick_lift_dense':
            shape_fn = lambda x: np.select([x > -0.3], [10 * (x + 0.3) ** 2 + x], x)
            offset = np.zeros_like(hand_pos)
            offset[:, -1] = - 0.03
            offset[:, :2] = (gripper_state[:, :2] + gripper_state[:, 2:]) / 2
            _ = obj_pos - (hand_pos + offset)
            touch_distances = np.linalg.norm(_, axis=1)
            touch_distances_xy = np.linalg.norm(_[:, :2], axis=1)
            gripper_opening = np.linalg.norm(gripper_state[:, :2] - gripper_state[:, 2:])
            dh = 0.083 - hand_pos[:, -1]
            opening = 0.02
            height_penalty = (-10 if touch_distances_xy > opening else 0) * np.maximum(dh, 0)
            grip_reward = (5 * (0.1 - gripper_opening)) if touch_distances_xy < opening and (
                    hand_pos[:, -1] - obj_pos[:, -1]) < 0.065 else ((gripper_opening - 0.1) * 5)
            # add all rewards
            # height_goal = 0.4
            height_goal = obj_goals[:, -1]
            r = shape_fn(-touch_distances)
            r += height_penalty
            r += grip_reward
            r -= mocap_error * 10  # add mocap error to avoid collision
            r += np.select([obj_pos[:, -1] > 0.025],  # okay b/c [:, -1] is scalar.
                           [0.5 - np.abs(obj_pos[:, -1] - height_goal)]
                           , 0) * 10
        elif reward_type == 'pick_reach_sparse':
            r = (np.linalg.norm(obj_goals - obj_pos, axis=-1) < 0.02).astype(float)
        elif reward_type == 'pick_reach_dense':
            shape_fn = lambda x: np.select([x > -0.3], [10 * (x + 0.3) ** 2 + x], x)
            offset = np.zeros_like(hand_pos)
            offset[:, -1] = - 0.03
            offset[:, :2] = (gripper_state[:, :2] + gripper_state[:, 2:]) / 2
            _ = obj_pos - (hand_pos + offset)
            touch_distances = np.linalg.norm(_, axis=1)
            touch_distances_xy = np.linalg.norm(_[:, :2], axis=1)
            gripper_opening = np.linalg.norm(gripper_state[:, :2] - gripper_state[:, 2:])
            dh = 0.083 - hand_pos[:, -1]
            opening = 0.02
            height_penalty = (-10 if touch_distances_xy > opening else 0) * np.maximum(dh, 0)
            grip_reward = (5 * (0.1 - gripper_opening)) if touch_distances_xy < opening and (
                    hand_pos[:, -1] - obj_pos[:, -1]) < 0.065 else ((gripper_opening - 0.1) * 5)
            # add all rewards
            r = shape_fn(-touch_distances)
            r += height_penalty
            r += grip_reward
            r -= mocap_error * 10  # add mocap error to avoid collision
            r += np.select([obj_pos[:, -1] > 0.025],
                           [0.5 - np.linalg.norm(obj_goals - obj_pos, axis=-1)],
                           0) * 10
            # r -= 6
            # # note: correction factor to match door reward with pick and place reward.
            # r = (r + 3.7) / 2.8 * 0.45
        # end of Ge's shaped rewards
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    @staticmethod
    def door_rewards(reward_type, actions, obs, index):
        hand_pos = obs['hand_pos']
        door_poses = obs['door_poses']
        hand_door_delta = obs['hand_door_delta']
        gr = obs['gripper']
        gripper_pos = obs['gripper_pos']  # only 3-dimensional

        i = 3 * index
        delta = hand_door_delta[:, i:i + 3] + [0, 0, 0.035]
        reach_distance = np.linalg.norm(delta, axis=-1)

        left_prong_delta = delta.copy()
        left_prong_delta[:, :2] -= gr[:, :2]
        offset_reach_distance = np.linalg.norm(
            left_prong_delta + ([-0.05, 0.1, 0.02] if left_prong_delta[:, 1] > -0.002 else [0, 0, -0.03]), axis=-1)
        y_targ = door_poses[:, 2 * index + 1] - 0.76

        mocap_error = np.linalg.norm(hand_pos - gripper_pos - [0, 0, 0.035], axis=-1)

        me_coef = 10  # scale the mocap error by 10 to match y_targ.

        if reward_type == 'reach_distance':
            reward = - reach_distance - mocap_error * me_coef
        elif reward_type == 'open_sparse':
            reward = (y_targ < - 0.09).astype(float)
        elif reward_type == 'open_dense':
            # The reach distance would get worse and the robot slides to the left to pull
            # on the door. However it is compensated by the opening of the door.
            reward = - y_targ - offset_reach_distance - mocap_error * me_coef
        else:
            raise NotImplemented
        return reward

    # method for meta_rl_tasks
    def sample_task(self, index=None):
        return self.controls.sample_task(index=index)

    # method for meta_rl_tasks
    def get_goal_index(self):
        return self.controls.index

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'hand_distance',
            'obj_distance',
            'hand_and_obj_distance',
            'touch_distance',
            'hand_success',
            'obj_success',
            'hand_and_obj_success',
            'touch_success',
        ]:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s%s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'Final %s%s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
            ))
        return statistics

    def get_env_state(self):
        base_state = super().get_env_state()
        goal = self._state_goal.copy()
        return base_state, goal

    def set_env_state(self, state):
        base_state, goal = state
        super().set_env_state(base_state)
        self._state_goal = goal


def mixed_env(**kwargs):
    from .flat_goal_env import FlatGoalEnv
    return FlatGoalEnv(SawyerMixedMultitaskEnv(**kwargs),
                       obs_keys=('hand_pos',
                                 'hand_dot',
                                 'obj_poses',
                                 'door_poses',
                                 'hand_obj_delta',
                                 'hand_door_delta',
                                 "obj_goal_delta",
                                 'gripper'),
                       goal_keys=tuple())


def limited_obs_debug(**kwargs):
    from .flat_goal_env import FlatGoalEnv
    return FlatGoalEnv(SawyerMixedMultitaskEnv(**kwargs),
                       obs_keys=(
                           'hand_pos',
                           'hand_dot',
                           'obj_poses',
                           # 'door_poses',
                           'hand_obj_delta',
                           # 'hand_door_delta',
                           "obj_goal_delta",
                           'gripper'
                       ),
                       goal_keys=tuple())


def mixed_env_with_id(**kwargs):
    from .flat_goal_env import FlatGoalEnv
    return FlatGoalEnv(SawyerMixedMultitaskEnv(**kwargs),
                       obs_keys=('hand_pos',
                                 'hand_dot',
                                 'obj_poses',
                                 'door_poses',
                                 'hand_obj_delta',
                                 'hand_door_delta',
                                 "obj_goal_delta",
                                 'gripper',
                                 'task_id'
                                 ),
                       goal_keys=tuple())


from gym.envs import register

register(
    id="Mixed-v0",
    entry_point=SawyerMixedMultitaskEnv,
    # Block goal can be in the air.
    kwargs=dict(frame_skip=5,
                k_tasks=6,
                reward_type="dense",
                obj_init_pos=reduce(iadd, [
                    (-0.2, 0.525, 0.05),
                    (-0.0, 0.525, 0.05),
                    (0.2, 0.525, 0.05)
                ], []),
                mocap_low=(-0.5, 0.25, 0.035),
                mocap_high=(0.5, 0.8, 0.35),
                obj_low=(-0.05, 0.35, 0.2),
                obj_high=(0.05, 0.35, 0.2)
                ),
    max_episode_steps=100,
    reward_threshold=-3.75,
)
# register(
#     id="MixedId-v0",
#     entry_point=mixed_env_with_id,
#     # Block goal can be in the air.
#     kwargs=dict(frame_skip=5,
#                 k_tasks=6,
#                 sample_task_on_reset=True,  # note: this is what shuffles the tasks
#                 reward_type="dense",
#                 mocap_low=(-0.5, 0.25, 0.035),
#                 mocap_high=(0.5, 0.8, 0.35),
#                 obj_low=(0.05, 0.45, 0.2),
#                 obj_high=(0.35, 0.6, 0.2)
#                 ),
#     max_episode_steps=100,
#     reward_threshold=-3.75,
# )
# obj_goal_low = (-0.05, 0.35, 0.2)
# obj_goal_high = (0.05, 0.35, 0.2)
# register(
#     id="MixedFixedMultitaskId-v0",
#     entry_point=mixed_env_with_id,
#     # Block goal can be in the air.
#     kwargs=dict(frame_skip=5,
#                 k_tasks=6,
#                 sample_task_on_reset=True,  # note: this is what shuffles the tasks
#                 reward_type="dense",
#                 obj_init_pos=reduce(iadd, [
#                     (-0.2, 0.525, 0.05),
#                     (-0.0, 0.525, 0.05),
#                     (0.2, 0.525, 0.05)
#                 ], []),
#                 mocap_low=(-0.5, 0.25, 0.035),
#                 mocap_high=(0.5, 0.8, 0.35),
#                 obj_low=obj_goal_low,
#                 obj_high=obj_goal_high
#                 ),
#     max_episode_steps=100,
#     reward_threshold=-3.75,
# )
# register(
#     id="MixedPointMultitask-v0",
#     entry_point=mixed_env,
#     # Block goal can be in the air.
#     kwargs=dict(frame_skip=5,
#                 k_tasks=6,
#                 reward_type="point_dense",
#                 obj_init_pos=reduce(iadd, [
#                     (-0.2, 0.525, 0.05),
#                     (-0.0, 0.525, 0.05),
#                     (0.2, 0.525, 0.05)
#                 ], []),
#                 mocap_low=(-0.5, 0.25, 0.035),
#                 mocap_high=(0.5, 0.8, 0.35),
#                 obj_low=obj_goal_low,
#                 obj_high=obj_goal_high
#                 ),
#     max_episode_steps=100,
#     reward_threshold=-3.75,
# )
# register(
#     id="MixedMultitaskDebug-v0",
#     entry_point="rl_maml_tf.envs.sawyer.mixed_env:limited_obs_debug",
#     # Block goal can be in the air.
#     kwargs=dict(frame_skip=5,
#                 k_tasks=6,
#                 reward_type="dense",
#                 mocap_low=(-0.05, 0.35, 0.035),
#                 mocap_high=(0.45, 0.7, 0.35),
#                 obj_low=(0.05, 0.45, 0.2),
#                 obj_high=(0.35, 0.6, 0.2)
#                 ),
#     max_episode_steps=100,
#     reward_threshold=-3.75,
# )
# register(
#     id="DoorMultitask-v0",
#     entry_point="rl_maml_tf.envs.sawyer.door_multitask:door_env",
#     # Block goal can be in the air.
#     kwargs=dict(frame_skip=5,
#                 k_doors=3,
#                 reward_type="door_dense",
#                 mocap_low=(-0.5, 0.25, 0.035),
#                 mocap_high=(0.5, 0.8, 0.35),
#                 ),
#     max_episode_steps=100,
#     reward_threshold=-3.75,
# )
# register(
#     id="DoorMultitaskSparse-v0",
#     entry_point="rl_maml_tf.envs.sawyer.door_multitask:door_env",
#     # Block goal can be in the air.
#     kwargs=dict(frame_skip=5,
#                 k_doors=3,
#                 reward_type="door_sparse",
#                 mocap_low=(-0.5, 0.25, 0.035),
#                 mocap_high=(0.5, 0.8, 0.35),
#                 ),
#     max_episode_steps=100,
#     reward_threshold=-3.75,
# )
