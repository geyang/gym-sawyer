"""
The sawyer robot starts in the center, because relying on starting from the
left is unreliable. We now shape the reward to encourage the agent to lift
the arm.

Entropy bonus does not work super well with these Sawyer environments. Exploration should
be done via other means, or provided by demonstrations.

                                                                   -- Ge
"""
from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict

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


class SawyerDoorMultitaskEnv(MultitaskEnv, SawyerXYZEnv, SawyerCamEnv):
    def __init__(
            self,
            k_tasks=1,
            num_doors=1,
            sample_task_on_reset=False,
            hand_init_pos=(0.2, 0.525, 0.2),
            reward_type="door_dense",
            cam_id=-1, width=84, height=84,
            **kwargs
    ):
        self.num_doors = num_doors
        self.controls = Controls(k_tasks=k_tasks)

        model_name = get_asset_full_path(f'sawyer_door-{self.num_doors}.xml')

        MultitaskEnv.__init__(self)
        SawyerXYZEnv.__init__(self, model_name=model_name, **kwargs)
        SawyerCamEnv.__init__(self, cam_id=cam_id, width=width, height=height)

        self.sample_task_on_reset = sample_task_on_reset
        self.reward_type = reward_type
        self.hand_init_pos = None if hand_init_pos is None else np.array(hand_init_pos)
        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
        )
        self.hand_space = Box(self.mocap_low, self.mocap_high)
        self.door_space = Box(np.array([0, 0] * self.controls.k),
                              np.array([1, 1] * self.controls.k))
        self.hand_door_delta_space = Box(np.array([-1] * self.controls.k * 3),
                                         np.array([1] * self.controls.k * 3))
        self.gripper_space = Box(np.array([-0.03] * 4), np.array([0.03] * 4), dtype=float)
        # self.task_space = Discrete(k_tasks)
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
            door_poses=self.door_space,  # the x, y position of the door sites
            hand_dot=self.hand_space,
            hand_door_delta=self.hand_door_delta_space,
            gripper=self.gripper_space,
            gripper_pos=self.hand_space,
            task_id=self.task_space,
        ))

    def viewer_setup(self, cam_id=None):
        SawyerCamEnv.viewer_setup(self, cam_id)

        camera = self.viewer.cam

        camera.lookat[0] = 0
        camera.lookat[1] = .65
        camera.lookat[2] = 0.2
        camera.distance = 1.6
        camera.elevation = -15
        camera.azimuth = 135

    def step(self, action):
        self.set_xyz_action(action[:3])
        self.do_simulation((action[3], -action[3]))
        ob = self._get_obs()
        reward = self.compute_reward(action, ob)
        info = self._get_info(ob)
        done = False
        return ob, reward, done, info

    # LOL why not????
    task_ids = np.eye(3, 3, )

    def _get_obs(self):
        e = self.get_endeff_pos()
        doors = self.get_door_poses()
        dot = self.get_endeff_vel()
        l, r = self.get_gripper_pos()

        return dict(
            hand_pos=e,
            door_poses=np.concatenate([[x, y] for x, y, z in doors]),
            hand_dot=dot,
            hand_door_delta=np.concatenate(doors - e),
            gripper=np.concatenate([l[:2] - e[:2], r[:2] - e[:2]]),
            gripper_pos=np.mean([l, r], axis=0),
            task_id=self.task_ids[self.controls.index],
        )

    def _get_info(self, ob):
        door_poses = ob['door_poses']
        y_targ = door_poses[2 * self.controls.index + 1] - 0.76
        return dict(success=y_targ < - 0.09)

    def get_door_poses(self):
        return [self.data.get_site_xpos(f"door_{i}_handle") for i in range(self.controls.k)]

    def _set_goal_marker(self):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        # note: add offset for object goal
        self.data.site_xpos[self.model.site_name2id('obj-goal-site')] = (self._state_goal)
        if self.hide_goal_markers:
            self.data.site_xpos[self.model.site_name2id('obj-goal-site'), 2] = (-1000)

    def _set_hand_goal(self, goal):
        """debugging function"""
        self.data.site_xpos[self.model.site_name2id('hand-goal-site')] = (goal)

    def _set_obj_goal(self, goal):
        """debugging function"""
        self.data.site_xpos[self.model.site_name2id('obj-goal-site')] = (goal)

    def _set_obj_xyz(self, pos):  # 3 * k objects
        assert len(pos) == 3, 'need to be 3D'
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def _set_obj_distractor_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        for i in range(self.controls.k):
            qpos[9 + i * 7:9 + i * 7 + 3] = pos[i * 3: i * 3 + 3]
        qvel[9:9 + len(pos) * 2] = 0
        self.set_state(qpos, qvel)

    def sample_goals(self, batch_size, p_obj_in_hand=0.5):
        return dict(door_index=[self.controls.index])

    def set_task(self, id):
        self.controls.index = id

    def reset_model(self):
        if self.sample_task_on_reset:
            self.controls.sample_task()
        self._reset_hand()
        # self._reset_doors()
        return self._get_obs()

    def _reset_doors(self):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:9 + self.controls.k + 1] = 0
        qvel[9:9 + self.controls.k + 1] = 0
        self.set_state(qpos, qvel)
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
        hand_pos = obs['hand_pos']
        door_poses = obs['door_poses']
        hand_door_delta = obs['hand_door_delta']
        gr = obs['gripper']
        gripper_pos = obs['gripper_pos']  # only 3-dimensional

        i = 3 * self.controls.index
        delta = hand_door_delta[:, i:i + 3] + [0, 0, 0.035]
        reach_distance = np.linalg.norm(delta, axis=-1)

        left_prong_delta = delta.copy()
        left_prong_delta[:, :2] -= gr[:, :2]
        offset_reach_distance = np.linalg.norm(
            left_prong_delta + ([-0.05, 0.1, 0.02] if left_prong_delta[:, 1] > -0.002 else [0, 0, -0.03]), axis=-1)
        y_targ = door_poses[:, 2 * self.controls.index + 1] - 0.76

        mocap_error = np.linalg.norm(hand_pos - gripper_pos - [0, 0, 0.035], axis=-1)

        me_coef = 10  # scale the mocap error by 10 to match y_targ.

        if self.reward_type == 'reach_distance':
            reward = - np.linalg.norm(delta + [0, 0.04, - 0.035], axis=-1) - mocap_error * me_coef
        elif self.reward_type == 'door_sparse':
            reward = (y_targ < - 0.09).astype(float)
        elif self.reward_type == 'door_dense':
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
        self._set_goal_marker()

    def render(self, mode, **kwargs):
        if mode == "glamor":
            # self.sim.model.light_active[:2] = False
            # self.sim.model.light_active[2:] = True
            mode = "rgb"
        return SawyerXYZEnv.render(self, mode, **kwargs)


def door_env(**kwargs):
    from .flat_goal_env import FlatGoalEnv
    return FlatGoalEnv(SawyerDoorMultitaskEnv(**kwargs),
                       obs_keys=('hand_pos', 'door_poses', 'hand_dot', 'hand_door_delta', 'gripper'),
                       goal_keys=('door_poses',))


def door_with_task_id(**kwargs):
    from .flat_goal_env import FlatGoalEnv
    return FlatGoalEnv(SawyerDoorMultitaskEnv(**kwargs),
                       obs_keys=('hand_pos', 'door_poses', 'hand_dot', 'hand_door_delta', 'gripper', 'task_id'),
                       goal_keys=('door_poses',))


from gym.envs import register

register(
    id="Door-v0",
    entry_point=door_env,
    # Block goal can be in the air.
    kwargs=dict(frame_skip=5,
                k_tasks=1,
                reward_type="reach_distance",
                mocap_low=(-0.1, 0.4, 0.05),
                mocap_high=(0.1, 0.6, 0.35),
                ),
    # max_episode_steps=100,
    # reward_threshold=-3.75,
)
# register(
#     id="DoorReach-v0",
#     entry_point=door_env,
#     # Block goal can be in the air.
#     kwargs=dict(frame_skip=5,
#                 k_tasks=1,
#                 reward_type="reach_distance",
#                 mocap_low=(-0.35, 0.35, 0.05),
#                 mocap_high=(0.35, 0.7, 0.35),
#                 ),
#     max_episode_steps=100,
#     reward_threshold=-3.75,
# )
# register(
#     id="DoorFixedSingleTask-v0",
#     entry_point=door_env,
#     # Block goal can be in the air.
#     kwargs=dict(frame_skip=5,
#                 k_tasks=1,
#                 reward_type="door_dense",
#                 hand_init_pos=(0, 0.525, 0.2),
#                 mocap_low=(-0.35, 0.35, 0.05),
#                 mocap_high=(0.35, 0.7, 0.35),
#                 ),
#     max_episode_steps=100,
#     reward_threshold=-3.75,
# )
# register(
#     id="DoorSingleTask-v0",
#     entry_point=door_env,
#     # Block goal can be in the air.
#     kwargs=dict(frame_skip=5,
#                 k_tasks=1,
#                 reward_type="door_dense",
#                 mocap_low=(-0.35, 0.35, 0.05),
#                 mocap_high=(0.35, 0.7, 0.35),
#                 ),
#     max_episode_steps=100,
#     reward_threshold=-3.75,
# )
# register(
#     id="DoorFixedMultitask-v0",
#     entry_point=door_env,
#     # Block goal can be in the air.
#     kwargs=dict(frame_skip=5,
#                 k_tasks=3,
#                 reward_type="door_dense",
#                 hand_init_pos=(0, 0.525, 0.2),
#                 mocap_low=(-0.35, 0.35, 0.05),
#                 mocap_high=(0.35, 0.7, 0.35),
#                 ),
#     max_episode_steps=100,
#     reward_threshold=-3.75,
# )
# register(
#     id="DoorFixedMultitaskSparse-v0",
#     entry_point=door_env,
#     # Block goal can be in the air.
#     kwargs=dict(frame_skip=5,
#                 k_tasks=3,
#                 reward_type="door_sparse",
#                 hand_init_pos=(0, 0.525, 0.2),
#                 mocap_low=(-0.35, 0.35, 0.05),
#                 mocap_high=(0.35, 0.7, 0.35),
#                 ),
#     max_episode_steps=100,
#     reward_threshold=-3.75,
# )
# register(
#     id="SawyerDoorMultitask-v0",
#     entry_point=door_env,
#     # Block goal can be in the air.
#     kwargs=dict(frame_skip=5,
#                 k_tasks=3,
#                 reward_type="door_dense",
#                 mocap_low=(-0.35, 0.35, 0.05),
#                 mocap_high=(0.35, 0.7, 0.35),
#                 ),
#     max_episode_steps=100,
#     reward_threshold=-3.75,
# )
# register(
#     id="SawyerDoorMultitaskSparse-v0",
#     entry_point=door_env,
#     # Block goal can be in the air.
#     kwargs=dict(frame_skip=5,
#                 k_tasks=3,
#                 reward_type="door_sparse",
#                 mocap_low=(-0.35, 0.35, 0.05),
#                 mocap_high=(0.35, 0.7, 0.35),
#                 ),
#     max_episode_steps=100,
#     reward_threshold=-3.75,
# )
# register(
#     id="SawyerDoorReachMultitask-v0",
#     entry_point=door_env,
#     # Block goal can be in the air.
#     kwargs=dict(frame_skip=5,
#                 k_tasks=3,
#                 reward_type="reach_distance",
#                 mocap_low=(-0.35, 0.35, 0.05),
#                 mocap_high=(0.35, 0.7, 0.35),
#                 ),
#     max_episode_steps=100,
#     reward_threshold=-3.75,
# )
# register(
#     id="SawyerDoorFixedMultitaskId-v0",
#     entry_point="rl_maml_tf.envs.sawyer.door_multitask:door_with_task_id",
#     # Block goal can be in the air.
#     kwargs=dict(frame_skip=5,
#                 k_tasks=3,
#                 sample_task_on_reset=True,
#                 reward_type="door_dense",
#                 hand_init_pos=(0, 0.525, 0.2),
#                 mocap_low=(-0.35, 0.35, 0.05),
#                 mocap_high=(0.35, 0.7, 0.35),
#                 ),
#     max_episode_steps=100,
#     reward_threshold=-3.75,
# )
# register(
#     id="SawyerDoorMultitaskId-v0",
#     entry_point=door_with_task_id,
#     # Block goal can be in the air.
#     kwargs=dict(frame_skip=5,
#                 k_tasks=3,
#                 sample_task_on_reset=True,
#                 reward_type="door_dense",
#                 mocap_low=(-0.35, 0.35, 0.05),
#                 mocap_high=(0.35, 0.7, 0.35),
#                 ),
#     max_episode_steps=100,
#     reward_threshold=-3.75,
# )
