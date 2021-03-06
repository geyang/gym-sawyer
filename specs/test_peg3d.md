
# Sawyer Peg3D Environment
We include the following domains in this test:

| Name     |         goal_keys   |              Action Space |           Observation Space |
|----------|---------------------|---------------------------|-----------------------------|
| Peg3D-v0 | "hand"              |                       nan |                         nan |

## sawyer:Peg3D-v0


```python
env = gym.make("sawyer:Peg3D-v0")
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
env = gym.make("sawyer:Peg3D-v0")
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

<img style="align-self:center;" src="videos/peg3d_test.gif" />
