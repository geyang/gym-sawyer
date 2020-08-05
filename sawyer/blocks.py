from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict

import mujoco_py
from .env_util import get_stat_in_paths, create_stats_ordered_dict, get_asset_full_path
from .multitask_env import MultitaskEnv
from .base import SawyerXYZEnv, SawyerCamEnv, GoalReaching

# ALL_TASKS = [
#     "pick",
#     "pick_place",
#     "stack"
# ]

geom_types = {None: 5, 'cylinder': 5, 'box': 6}
geom_xy = {None: 1, 'cylinder': 1, 'box': 2}


class SawyerPickAndPlaceEnv(GoalReaching, SawyerXYZEnv, SawyerCamEnv):

    def __init__(
            self,
            task=None,
            num_objs=2,
            obj_low=None,
            obj_high=None,
            obj_type=None,
            obj_size=None,
            init_mode=None,
            gripper=None,
            fixed_goal=None,
            cam_id=-1,
            **kwargs
    ):
        self.task = task
        self.num_objs = num_objs

        model_name = get_asset_full_path(f'sawyer_pick_and_place-{num_objs}.xml')

        SawyerXYZEnv.__init__(self, model_name=model_name, **kwargs)
        SawyerCamEnv.__init__(self, cam_id=cam_id, **kwargs)

        # self.obj_size = obj_size
        # 37 (the last one out of 38) is the first object, and so on.
        if obj_type:
            # self.model.geom_type[37: 37 + self.num_objs, :2] = obj_type
            self.model.geom_type[37: 37 + self.num_objs] = geom_types[obj_type]
        if obj_size:
            self.model.geom_size[37: 37 + self.num_objs, :geom_xy[obj_type]] = obj_size

        self.init_mode = init_mode
        self.gripper_init = gripper

        self.fixed_goal = None if fixed_goal is None else np.array(fixed_goal)
        self._state_goal = None

        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
        )
        self.hand_obj_space = Box(
            np.concatenate((self.mocap_low, self.mocap_low)),
            np.concatenate((self.mocap_high, self.mocap_high)),
        )
        self.hand_obj_dot_space = Box(
            np.concatenate((self.mocap_low, self.mocap_low, obj_low)),
            np.concatenate((self.mocap_high, self.mocap_high, obj_high)),
        )
        self.hand_space = Box(self.mocap_low, self.mocap_high)
        self.obj_space = Box(np.array(obj_low), np.array(obj_high))
        self.gripper_space = Box(np.array([-0.03] * 4), np.array([0.03] * 4), dtype=float)
        """
        The gym reacher environment returns the following.
            cos(theta), sin(theta), goal, theta_dot, delta
        Here instead, we return
            cos(theta), sin(theta), theta_dot, goal_posts (target + distractor), delta (with fingertip)
        We don't ever return the position of the finger tip.
        """
        self.observation_space = Dict([
            ('desired_goal', self.hand_obj_space),
            ('achieved_goal', self.hand_obj_space),
            ('observation', self.hand_obj_dot_space),
            ('state_observation', self.hand_obj_dot_space),
            ('state_desired_goal', self.hand_obj_space),
            ('state_achieved_goal', self.hand_obj_space),
            ('state_delta', self.hand_obj_space),
            ('state_touch_distance', self.hand_space),
            ('state_gripper', self.gripper_space),
        ])

    def viewer_setup(self, cam_id=None):

        camera = self.viewer.cam

        camera.trackbodyid = -2
        camera.lookat[0] = 0
        camera.lookat[1] = .5
        camera.lookat[2] = 0.2
        camera.distance = 1.6
        camera.elevation = -25
        camera.azimuth = -45

    # used to produce flattened vector ob['x']
    FLAT_KEYS = "hand", "gripper", "obj_0"

    def state_dict(self):
        obj_poses = {f"obj_{i}": self.get_obj_pos(i) for i in range(self.num_objs)}
        return dict(
            hand=self.get_endeff_pos(),
            gripper=self.get_gripper_state(),
            **obj_poses
        )

    def get_obj_pos(self, obj_id=0):
        return self.data.get_body_xpos(f'obj_{obj_id}').copy()

    def _set_obj_xyz(self, pos, obj_id=0):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        offset = obj_id * 7
        qpos[9 + offset:12 + offset] = pos.copy()
        # offset = obj_id * 6
        # qvel[9 + offset:12 + offset] = 0
        self.set_state(qpos, qvel)

    def _set_bodies(self):
        if self.num_objs > 1:
            self._set_obj_xyz(self.obj_pos_2, obj_id=1)

    obj_pos_2 = np.array([0.0, 0.5, 0.02])

    def reset_model(self, mode=None):
        """Provide high-level `mode` for sampling types of goal configurations."""
        self.sim.reset()
        mode = mode or self.init_mode

        if self.num_objs > 1:
            self._set_obj_xyz(self.obj_pos_2, obj_id=1)
        # always drop object 0 from the air.
        obj_pos = self.obj_space.sample()
        self._set_obj_xyz(obj_pos)
        print("object:", obj_pos, self.obj_space)

        rd = self.np_random.rand()
        if mode is None:
            if rd < 0.45:
                mode = 'hover'
            elif rd < 0.55:
                mode = "pick"
            elif rd < 0.9:
                mode = "in-hand-hover"
            else:
                mode = "on-top"

        if mode == 'hover':  # hover
            hand_pos = self.hand_space.sample()
            print("hand", hand_pos, self.hand_space)
            # note: this is the only free wavering mode
            if self.gripper_init is not None:
                self.gripper = self.gripper_init
            else:
                self.gripper = self.np_random.choice([-1, 1], size=1)
            self._reset_hand(hand_pos, [self.gripper, -self.gripper])
        elif mode == "pick":
            # self.gripper = self.np_random.choice([1, -1], size=1)
            self.gripper = -1
            obj_pos[-1] = self.hand_space.sample()[-1]
            self._reset_hand(obj_pos, [self.gripper, -self.gripper])
        elif mode == "in-hand-hover":
            hand_pos = self.hand_space.sample()
            # hand_pos[-1] = np.max([0.08, hand_pos[-1]])
            # info: make sure in-hand-hover is in the air.
            hand_pos[-1] = 0.15
            self._reset_hand(hand_pos)
            self.put_obj_in_hand()
        elif mode == 'on-top':
            pos = self.obj_pos_2.copy()
            pos[-1] = np.max([0.08, self.hand_space.sample()[-1]])
            self._reset_hand(pos)
            self.put_obj_in_hand()
        else:
            raise NotImplementedError(f"{mode} is not supported")

        return self.get_obs()

    def put_obj_in_hand(self, obj_id=0):
        new_obj_pos = self.data.get_site_xpos('endEffector')
        new_obj_pos[1] -= 0.01
        self.do_simulation([-1, 1])
        self._set_obj_xyz(new_obj_pos, obj_id)
        self.gripper = 1
        self.do_simulation([self.gripper, -self.gripper])

    """
    Multitask functions
    """

    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }

    def sample_goals(self, batch_size, p_obj_in_hand=0.5):
        if self.fixed_goal is not None:
            goals = np.repeat(self.fixed_goal.copy()[None], batch_size, 0)
        else:
            goals = self.np_random.uniform(
                self.hand_obj_space.low,
                self.hand_obj_space.high,
                size=(batch_size, self.hand_obj_space.low.size),
            )
        # num_objs_in_hand = int(batch_size * p_obj_in_hand)
        # # Put object in hand
        # goals[:num_objs_in_hand, 3:] = goals[:num_objs_in_hand, :3].copy()
        # goals[:num_objs_in_hand, 4] -= 0.01
        # # Put object one the table (not floating)
        # goals[num_objs_in_hand:, 5] = self.obj_init_pos[2]
        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def compute_rewards(self, actions, obs):
        return [0] * len(actions)

    def get_env_state(self):
        base_state = super().get_env_state()
        goal = self._state_goal.copy()
        return base_state, goal

    def set_env_state(self, state):
        base_state, goal = state
        super().set_env_state(base_state)
        self._state_goal = goal
        # self._set_goal_marker()

    def render(self, mode, **kwargs):
        if mode == "glamor":
            self.sim.model.light_active[:2] = False
            self.sim.model.light_active[2:] = True
            mode = "rgb"
        return SawyerXYZEnv.render(self, mode, **kwargs)


# used for state space baselines
def pick_place_env(**kwargs):
    from .flat_goal_env import FlatGoalEnv
    return FlatGoalEnv(SawyerPickAndPlaceEnv(**kwargs),
                       obs_keys=('state_observation', 'state_desired_goal',
                                 'state_delta', 'state_touch_distance', 'state_gripper'),
                       goal_keys=('state_desired_goal',))


from gym.envs import register

register(
    id="PickPlace-v0",
    entry_point=SawyerPickAndPlaceEnv,
    # Place goal has to be on the surface.
    kwargs=dict(frame_skip=5,
                # reward_type="pick_place_dense",
                mocap_low=(-0.1, 0.45, 0.05),
                mocap_high=(0.1, 0.55, 0.22),
                obj_low=(-0.1, 0.425, 0.08),
                obj_high=(0.1, 0.525, 0.08)
                ),
    # max_episode_steps=100,
    # reward_threshold=-3.75,
)
register(
    id="Push-v0",
    entry_point=SawyerPickAndPlaceEnv,
    # Place goal has to be on the surface.
    kwargs=dict(frame_skip=5,
                # reward_type="pick_place_dense",
                init_mode="hover",
                gripper=1,
                obj_type="cylinder",
                obj_size=0.07,
                mocap_low=(-0.2, 0.7, 0.06),
                mocap_high=(0.2, 0.3, 0.05),
                obj_low=(-0.15, 0.65, 0.08),
                obj_high=(0.15, 0.35, 0.08)
                ),
    # max_episode_steps=100,
    # reward_threshold=-3.75,
)
