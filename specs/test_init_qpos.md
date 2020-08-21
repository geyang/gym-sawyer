
# Sawyer Peg3D Environment
We include the following domains in this test:

| Name     |         goal_keys   |              Action Space |           Observation Space |
|----------|---------------------|---------------------------|-----------------------------|
| Peg3D-v0 | "hand"              |                       nan |                         nan |

## sawyer:Peg3D-v0


```python
env = gym.make("sawyer:Reach-v0")
env.reset()
```

<div style="flex-wrap:wrap; display:flex; flex-direction:row; item-align:center;"><div><div style="text-align: center">Reach-v0</div><img style="margin:0.5em;" src="data:image/png;base64,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" /></div></div>