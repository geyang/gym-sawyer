
# Sawyer Push Task (Reward)

Relabeling using the sparse reward does not help, because
much of the MDP is disentangled from the robot unless the
gripper is touching the object.

## Standard push environment

``` python
env = gym.make('sawyer:Push-v0', cam_id=0)
env = GoalImg(env)
env.seed(100)
obs = env.reset()
for step in range(100):
    delta = obs['goal'] - obs['x']
    act = delta[:4] * [5, 5, 0, 1]
    obs, reward, done, info = env.step(act)
    doc.image(obs['img'].transpose([1, 2, 0]), f"figures/sequence/push_step_{step}.png", caption=f"Step {step}")
    if np.linalg.norm(obs['x'][:3] - obs['goal'][:3]) < 0.01:
        break
doc.image(obs['goal_img'].transpose([1, 2, 0]), "figures/sequence/push_goal.png", caption="Goal Image")
```
<div style="flex-wrap:wrap; display:flex; flex-direction:row; item-align:center;"><div><div style="text-align: center">Step 0</div><img style="margin:0.5em;" src="figures/sequence/push_step_0.png" /></div><div><div style="text-align: center">Step 1</div><img style="margin:0.5em;" src="figures/sequence/push_step_1.png" /></div><div><div style="text-align: center">Step 2</div><img style="margin:0.5em;" src="figures/sequence/push_step_2.png" /></div><div><div style="text-align: center">Step 3</div><img style="margin:0.5em;" src="figures/sequence/push_step_3.png" /></div><div><div style="text-align: center">Step 4</div><img style="margin:0.5em;" src="figures/sequence/push_step_4.png" /></div><div><div style="text-align: center">Step 5</div><img style="margin:0.5em;" src="figures/sequence/push_step_5.png" /></div><div><div style="text-align: center">Step 6</div><img style="margin:0.5em;" src="figures/sequence/push_step_6.png" /></div><div><div style="text-align: center">Step 7</div><img style="margin:0.5em;" src="figures/sequence/push_step_7.png" /></div><div><div style="text-align: center">Step 8</div><img style="margin:0.5em;" src="figures/sequence/push_step_8.png" /></div><div><div style="text-align: center">Step 9</div><img style="margin:0.5em;" src="figures/sequence/push_step_9.png" /></div><div><div style="text-align: center">Step 10</div><img style="margin:0.5em;" src="figures/sequence/push_step_10.png" /></div><div><div style="text-align: center">Step 11</div><img style="margin:0.5em;" src="figures/sequence/push_step_11.png" /></div><div><div style="text-align: center">Step 12</div><img style="margin:0.5em;" src="figures/sequence/push_step_12.png" /></div><div><div style="text-align: center">Step 13</div><img style="margin:0.5em;" src="figures/sequence/push_step_13.png" /></div><div><div style="text-align: center">Step 14</div><img style="margin:0.5em;" src="figures/sequence/push_step_14.png" /></div><div><div style="text-align: center">Step 15</div><img style="margin:0.5em;" src="figures/sequence/push_step_15.png" /></div><div><div style="text-align: center">Step 16</div><img style="margin:0.5em;" src="figures/sequence/push_step_16.png" /></div><div><div style="text-align: center">Step 17</div><img style="margin:0.5em;" src="figures/sequence/push_step_17.png" /></div><div><div style="text-align: center">Step 18</div><img style="margin:0.5em;" src="figures/sequence/push_step_18.png" /></div><div><div style="text-align: center">Step 19</div><img style="margin:0.5em;" src="figures/sequence/push_step_19.png" /></div><div><div style="text-align: center">Step 20</div><img style="margin:0.5em;" src="figures/sequence/push_step_20.png" /></div><div><div style="text-align: center">Step 21</div><img style="margin:0.5em;" src="figures/sequence/push_step_21.png" /></div><div><div style="text-align: center">Step 22</div><img style="margin:0.5em;" src="figures/sequence/push_step_22.png" /></div><div><div style="text-align: center">Step 23</div><img style="margin:0.5em;" src="figures/sequence/push_step_23.png" /></div><div><div style="text-align: center">Step 24</div><img style="margin:0.5em;" src="figures/sequence/push_step_24.png" /></div><div><div style="text-align: center">Step 25</div><img style="margin:0.5em;" src="figures/sequence/push_step_25.png" /></div><div><div style="text-align: center">Step 26</div><img style="margin:0.5em;" src="figures/sequence/push_step_26.png" /></div><div><div style="text-align: center">Step 27</div><img style="margin:0.5em;" src="figures/sequence/push_step_27.png" /></div><div><div style="text-align: center">Step 28</div><img style="margin:0.5em;" src="figures/sequence/push_step_28.png" /></div><div><div style="text-align: center">Step 29</div><img style="margin:0.5em;" src="figures/sequence/push_step_29.png" /></div><div><div style="text-align: center">Goal Image</div><img style="margin:0.5em;" src="figures/sequence/push_goal.png" /></div></div>
``` python
for key, value in obs.items():
    doc.print(key, value.shape)
```
```
hand (3,)
```
```
gripper (1,)
```
```
obj_0 (3,)
```
```
x (7,)
```
```
goal (7,)
```
```
goal_img (3, 100, 100)
```
```
img (3, 100, 100)
```
