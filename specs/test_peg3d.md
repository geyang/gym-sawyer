
# Sawyer Peg3D Environment
We include the following domains in this test:

| Name     |         goal_keys   |              Action Space |           Observation Space |
|----------|---------------------|---------------------------|-----------------------------|
| Peg3D-v0 | "hand"              |                       nan |                         nan |

## sawyer:Peg3D-v0


```python
env = gym.make("sawyer:Peg3D-v0", cam_id=0)
```

Make sure that the spec agrees with what it returns

```python
doc.yaml(space2dict(env.observation_space))
```

```yaml
hand: shape(3,)
slot: shape(3,)
```

```python
doc.yaml(obs2dict(obs))
```

```yaml
hand: Shape(3,)
slot: Shape(3,)
```
<div style="flex-wrap:wrap; display:flex; flex-direction:row; item-align:center;"><div><div style="text-align: center">Peg3D-v0</div><img style="margin:0.5em;" src="data:image/png;base64,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" /></div></div>

To use with RL algorithms, use the FlatGoalEnv wrapper:


```python
env = gym.make("sawyer:Peg3D-v0", cam_id=0)
env = FlatGoalEnv(env)
```

Which gives the following observation type

```python
doc.yaml(space2dict(env.observation_space))
```

```yaml
shape(9,)
...
```

```python
doc.yaml(obs2dict(obs))
```

```yaml
Shape(9,)
...
```

## sawyer:Reach-v0


```python
env = gym.make("sawyer:Peg3D-v0")
```


<video width="320" height="240" controls="true">
  <source src="videos/peg3d_test.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

