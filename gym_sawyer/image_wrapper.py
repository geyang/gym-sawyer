from gym import Wrapper


class GoalImage(Wrapper):
    """
    Wrapper Class Create 84x84 RGBD viewers
    """

    def __init__(self, env):
        self.env = env

    def reset(self, **kwargs):
        self.env.reset(**kwargs)

    def _get_obs(self):
        pass

    def render(self, mode="notebook", width=None, height=None):
        pass
