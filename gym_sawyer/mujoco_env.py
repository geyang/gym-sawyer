import os
import numpy as np
from os import path
import gym
from gym import error, spaces
from gym.utils import seeding

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e))

DEFAULT_SIZE = 640, 480


class MujocoEnv(gym.Env):
    """
    This is a simplified version of the gym MujocoEnv class.

    Some differences are:
     - Do not automatically set the observation/action space.
    """

    def __init__(self, model_path, frame_skip=4, set_spaces=False):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        if set_spaces:
            observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
            assert not done
            self.obs_dim = observation.size

            bounds = self.model.actuator_ctrlrange.copy()
            low = bounds[:, 0]
            high = bounds[:, 1]
            self.action_space = spaces.Box(low=low, high=high)

            high = np.inf * np.ones(self.obs_dim)
            low = -high
            self.observation_space = spaces.Box(low, high)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def reset(self):
        self.sim.reset()
        ob = self.reset_model()
        old_viewer = self.viewer
        for v in self._viewers.values():
            self.viewer = v
            self.viewer_setup()
        self.viewer = old_viewer
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames=None):
        if n_frames is None:
            n_frames = self.frame_skip
        if self.sim.data.ctrl is not None and ctrl is not None:
            self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def _before_render(self):
        """
        setting site (markers) in mujoco have to happen

        - *after* a view object has been created, and
        - *before* rendering.

        This is why we add this hook here so that environments can add marker setting code
        in-between these two steps.

        :return:
        """
        pass

    def render(self, mode='human', width=None, height=None, cam_id=None):
        """
        returns images of modality <modeL

        :param mode: One of ['human', 'rgb', 'rgbd', 'depth']
        :param kwargs: width, height (in pixels) of the image.
        :return: image(, depth). image is between [0, 1), depth is distance.
        """
        width = width or self.default_window_width
        height = height or self.default_window_height
        viewer = self._get_viewer(mode)

        if cam_id is not None:
            self.viewer_setup(cam_id)

        self._before_render()

        # note-1: original image is upside-down, so flip it
        # note-2: depth channel is float, in meters. Not normalized.
        if mode in ['rgb', 'rgb_array']:
            viewer.render(width, height)
            data = viewer.read_pixels(width, height, depth=False)
            return data[::-1, :, :].astype(np.uint8)
        elif mode == 'rgbd':
            viewer.render(width, height)
            rgb, d = viewer.read_pixels(width, height, depth=True)
            return rgb[::-1, :, :], d[::-1, :]
        elif mode == 'depth':
            viewer.render(width, height)
            _, d = viewer.read_pixels(width, height, depth=True)
            return d[::-1, :]
        elif mode == 'grey':
            viewer.render(width, height)
            data = viewer.read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :].mean(axis=-1).astype(np.uint8)
        elif mode == "notebook":
            from IPython.display import display
            from PIL import Image

            viewer.render(width, height)
            data = viewer.read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            display(Image.fromarray(data[::-1, :, :]))

        elif mode == 'human':
            if width and height:
                import glfw
                glfw.set_window_size(viewer.window, width, height)
            viewer.render()

    def close(self):
        self.viewer = None

        for viewer in self._viewers.items():
            import glfw
            glfw.destroy_window(viewer.window)

        self._viewers.clear()

    default_window_width = DEFAULT_SIZE[0]
    default_window_height = DEFAULT_SIZE[1]

    def _get_viewer(self, mode) -> mujoco_py.MjViewer:
        self.viewer = self._viewers.get(mode)
        if self.viewer is not None:
            return self.viewer

        if mode == 'human':
            self.viewer = mujoco_py.MjViewer(self.sim)
            # we turn off the overlay and make the window smaller.
            self.viewer._hide_overlay = True
            import glfw
            glfw.set_window_size(self.viewer.window, self.default_window_width, self.default_window_height)
        else:
            self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)
        self.viewer_setup()
        self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])

    def get_image(self, width=84, height=84, camera_name=None):
        return self.sim.render(
            width=width,
            height=height,
            camera_name=camera_name,
        )

    # def initialize_camera(self, init_fctn):
    #     sim = self.sim
    #     viewer = mujoco_py.MjRenderContextOffscreen(sim, device_id=-1)
    #     init_fctn(viewer.cam)
    #     sim.add_render_context(viewer)
