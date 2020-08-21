import numpy as np
from gym import Wrapper, spaces
from gym.spaces import Box, Dict

from .env_util import get_asset_full_path
from .base import SawyerXYZEnv, SawyerCamEnv, pick

# ALL_TASKS = [
#     "pick",
#     "pick_place",
#     "stack"
# ]

geom_types = {None: 5, 'cylinder': 5, 'box': 6}
geom_xy = {None: 1, 'cylinder': 1, 'box': 2}


class SawyerObjEnv(SawyerCamEnv, SawyerXYZEnv):


    def __init__(
            self,
            task=None,
            num_objs=1,
            obj_low=None,
            obj_high=None,
            obj_type=None,
            obj_size=None,
            init_mode=None,
            goal_mode=None,
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
        self.goal_mode = goal_mode

        # extend the observation space with objects
        self.obj_space = Box(np.array(obj_low), np.array(obj_high))
        d = self.observation_space.spaces.copy()
        d = pick(d, *self.obs_keys)
        for n in range(num_objs):
            self.obs_keys = tuple({*self.obs_keys, f'obj_{n}'})
            d[f'obj_{n}'] = self.obj_space
        self.observation_space = spaces.Dict(**d)

    obs_keys = "hand", "obj_0"
    goal_keys = "obj_0",

    # override parent class
    def state_dict(self):
        arm_poses = super().state_dict()
        obj_poses = {f"obj_{i}": self.get_obj_pos(i) for i in range(self.num_objs)}
        return {**arm_poses, **obj_poses}

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

    def _set_markers(self):
        if self.num_objs > 1:
            self._set_obj_xyz(self.obj_pos_2, obj_id=1)

    obj_pos_2 = np.array([0.0, 0.5, 0.02])

    def put_obj_in_hand(self, obj_id=0):
        new_obj_pos = self.data.get_site_xpos('endEffector')
        new_obj_pos[1] -= 0.01
        self.do_simulation([-1, 1], n_frames=10)
        self._set_obj_xyz(new_obj_pos, obj_id)
        self.gripper = 1
        self.do_simulation([self.gripper, -self.gripper], n_frames=10)

    def reset_model(self, mode=None, obj_pos=None, hand_pos=None, to_goal=False):
        """Provide high-level `mode` for sampling types of goal configurations."""
        self._fast_reset()

        mode = mode or (self.goal_mode if to_goal else self.init_mode)

        if self.num_objs > 1:
            self._set_obj_xyz(self.obj_pos_2, obj_id=1)
        # always drop object 0 from the air.
        if obj_pos is None:
            obj_pos = self.obj_space.sample()
        self._set_obj_xyz(obj_pos)

        if hand_pos is not None:
            if self.gripper_init is not None:
                self.gripper = self.gripper_init
            else:
                self.gripper = self.np_random.choice([-1, 1], size=1)
            self._reset_hand(hand_pos, [self.gripper, -self.gripper])
            return self.get_obs()

        if mode is None:
            rd = self.np_random.rand()
            if rd < 0.45:
                mode = 'hover'
            elif rd < 0.55:
                mode = "pick"
            elif rd < 0.9:
                mode = "in-hand-hover"
            else:
                mode = "on-top"

        if mode == 'hover':  # hover
            hand_pos = self.hand_space.sample() + [0, 0, 0.06]
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

    # def render(self, mode, **kwargs):
    #     # if mode == "glamor":
    #     #     self.sim.model.light_active[:2] = False
    #     #     self.sim.model.light_active[2:] = True
    #     #     mode = "rgb"
    #     return SawyerXYZEnv.render(self, mode, **kwargs)

    # """
    # Multitask functions, Now deprecated
    # """
    # def get_env_state(self):
    #     base_state = super().get_env_state()
    #     goal = self._state_goal.copy()
    #     return base_state, goal
    #
    # def set_env_state(self, state):
    #     base_state, goal = state
    #     super().set_env_state(base_state)
    #     self._state_goal = goal
    #     # self._set_goal_marker()


def expand(a):
    if a is None or isinstance(a, str):
        return a
    elif isinstance(a, dict):
        return {k: expand(v) for k, v in a.items()}
    else:
        return [*a, 0]


class LowActDim(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        actsp = self.action_space
        self.action_space = spaces.Box(actsp.low[:-2], actsp.high[:-2])

    def reset_model(self, *args, **kwargs):
        return self.env.reset_model(*[expand(a) for a in args],
                                    **{k: expand(v) for k, v in kwargs.items()})

    def step(self, act):
        return self.env.step([*act, 0, 0])


def low_dim_push(**kwargs):
    return LowActDim(SawyerObjEnv(**kwargs))


from gym.envs import register

register(
    id="PickPlace-v0",
    entry_point=SawyerObjEnv,
    # Place goal has to be on the surface.
    kwargs=dict(frame_skip=5,
                # init_mode is None, using shaped init
                goal_mode="hover",
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
    entry_point=low_dim_push,
    # Place goal has to be on the surface.
    kwargs=dict(frame_skip=5,
                # init_mode is None, using shaped init
                goal_mode="hover",
                gripper=1,
                num_objs=1,
                obj_type="cylinder",
                obj_size=0.05,
                cam_id=0,
                mode="rgb",
                mocap_low=(-0.25, 0.3, 0.06),
                mocap_high=(0.25, 0.7, 0.06),
                obj_low=(-0.15, 0.35, 0.08),
                obj_high=(0.15, 0.65, 0.08),
                show_mocap=False
                ),
    # max_episode_steps=100,
    # reward_threshold=-3.75,
)
register(
    id="PushMove-v0",
    entry_point=low_dim_push,
    # Place goal has to be on the surface.
    kwargs=dict(frame_skip=5,
                # init_mode is None, using shaped init
                goal_mode="hover",
                goal_keys=("hand", "obj_0"),
                gripper=1,
                num_objs=1,
                obj_type="cylinder",
                obj_size=0.05,
                cam_id=0,
                mode="rgb",
                mocap_low=(-0.25, 0.3, 0.06),
                mocap_high=(0.25, 0.7, 0.06),
                obj_low=(-0.15, 0.35, 0.08),
                obj_high=(0.15, 0.65, 0.08),
                show_mocap=False
                ),
    # max_episode_steps=100,
    # reward_threshold=-3.75,
)
