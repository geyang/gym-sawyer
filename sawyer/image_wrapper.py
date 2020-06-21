from gym import Wrapper
import numpy as np


class GoalImage(Wrapper):
    """
    Wrapper Class Create 84x84 RGBD viewers
    """

    def __init__(self, env, depth_range=(0.983, 0.993)):
        self.env = env
        self.depth_range = depth_range

    def _rgbd(self, img, depth):
        drange = self.depth_range
        norm_depth = 255 * (depth - drange[0]) / (drange[1] - drange[0])
        rgbd = np.concatenate([img, norm_depth[..., None].astype(np.uint8)], axis=-1)
        return rgbd

    def reset(self, **kwargs):
        env = self.env

        obs = env.reset(**kwargs)
        self.put_obj_in_hand()
        img, depth = env.render("rgbd", **kwargs)
        obs['goal_img'] = self._rgbd(img, depth)

        return obs

    def step(self, action, **kwargs):
        obs, reward, done, info = self.env.step(action, **kwargs)
        return obs


if __name__ == '__main__':
    import gym
    from PIL import Image

    raw = gym.make("sawyer:PickPlace-v0", cam_id=0, show_mocap=False)
    env = GoalImage(raw)

    obs = env.reset()
    print(obs.keys())
    img = Image.fromarray(obs['goal_img'][:, :, :3])
    img.show()

    d_ch = obs['goal_img'][:, :, 3]
    img = Image.fromarray(d_ch, "L")
    print(f"max: {d_ch.max()}, min: {d_ch.min()}")
    img.show()

    print('hang in there')
