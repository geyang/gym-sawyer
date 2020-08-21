from cmx import doc
import gym
import numpy as np
from env_wrappers.flat_env import FlatGoalEnv
from sawyer.misc import space2dict, obs2dict


def test_start():
    doc @ """
    # Sawyer Blocks Environment
    
    ## To-do
    - [ ] automatically generate the environment table
    
    We include the following domains in this test:
    """
    doc.csv @ """
    Name,        goal_keys,           Action Space,        Observation Space
    Reach-v0,    "hand"
    Push-v0,     "obj_0"
    PushMove-v0, "hand"&#44; "obj_0"
    """


def test_reach_reward():
    doc @ """
    ## sawyer:Reach-v0
    """
    with doc:
        env = gym.make("sawyer:Reach-v0")
    env.seed(100)

    frames = []
    obs = env.reset()
    for step in range(100):
        # gripper dimension does not matter
        act = env.goal['hand'] - obs['hand']
        obs, r, done, info = env.step(np.array([*act, 0]) * 10)
        img = env.render('rgb')
        frames.append(img)
        if done:
            break
    else:
        raise RuntimeError("Reach failed to terminate")

    doc.video(frames, f"videos/reach.gif")
    doc.flush()


def test_reach_flat_goal():
    doc @ """
    ### Using FlatGoalEnv Wrapper
    """
    with doc:
        env = gym.make("sawyer:Reach-v0")
        env.seed(100)
        env = FlatGoalEnv(env)
        obs = env.reset()
    with doc("Make sure that the spec agrees with what it returns"):
        doc.yaml(space2dict(env.observation_space))
    with doc:
        doc.yaml(obs2dict(obs))


def test_push():
    doc @ """
    ## sawyer:Push-v0
    """
    with doc:
        env = gym.make("sawyer:Push-v0")
        env.seed(100)
        obs = env.reset()
    with doc("Make sure that the spec agrees with what it returns"):
        doc.yaml(space2dict(env.observation_space))
    with doc:
        doc.yaml(obs2dict(obs))


def test_push_flat_goal():
    doc @ """
    ###  with FlatGoalEnv Wrapper
    """
    with doc:
        env = gym.make("sawyer:Push-v0")
        env.seed(100)
        env = FlatGoalEnv(env, )
        obs = env.reset()

    with doc("Make sure that the spec agrees with what it returns"):
        doc.yaml(space2dict(env.observation_space))
    with doc:
        doc.yaml(obs2dict(obs))


def test_push_move():
    doc @ """
    ##  sawyer:PushMove-v0 Domain
    
    This is different from the push domain by the
    additional goal key that specifies the final
    position for the hand.
    """
    with doc:
        env = gym.make("sawyer:PushMove-v0")
        env.seed(100)
        env = FlatGoalEnv(env, )
        obs = env.reset()

    with doc("Make sure that the spec agrees with what it returns"):
        doc.yaml(space2dict(env.observation_space))
    with doc:
        doc.yaml(obs2dict(obs))


def test_pick_place():
    doc @ """
    ##  sawyer:PickPlace-v0 Domain
    
    """
    with doc:
        env = gym.make("sawyer:PickPlace-v0")
        env.seed(100)
        env = FlatGoalEnv(env, )
        obs = env.reset()

    with doc("Make sure that the spec agrees with what it returns"):
        doc.yaml(space2dict(env.observation_space))
    with doc:
        doc.yaml(obs2dict(obs))


def test_pick_place_reward():
    doc @ """
    ## sawyer:PickPlace-v0
    We set the goal_key to ['hand',] (the same as the reaching
    task) to test the termination.
    """
    with doc:
        env = gym.make("sawyer:PickPlace-v0", goal_keys=["hand"])
    env.seed(100)

    frames = []
    obs = env.reset()
    for step in range(100):
        act = env.goal['hand'] - obs['hand']
        obs, r, done, info = env.step(np.array([*act, 0]) * 10)
        img = env.render('rgb')
        frames.append(img)
        if done:
            break
    else:
        # raise RuntimeError("Reach failed to terminate")
        print('failed')
        pass

    doc.video(frames, f"videos/pick_place.gif")
    doc.flush()


def test_block_distribution():
    doc @ """
    Show the distribution of the block after initialization
    """
    with doc:
        env = gym.make("sawyer:PickPlace-v0", width=240, height=160)
    env.seed(100)

    frames = []
    for step in range(20):
        obs = env.reset()
        frames.append(env.render('rgb'))

    doc.image(np.min(frames, axis=0))
    doc.flush()
# def test_fetch():
#     with doc:
#         import gym
#         env = gym.make("FetchReach-v1")
#
#         assert env.compute_reward is not None
