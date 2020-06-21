# `gym-Sawyer`, a Sawyer plugin for OpenAI Gym

Current release at `v0.1.0`.

## Installation

The `gym-sawyer` package name is currently being squatted [(issue PEP541)](https://github.com/pypa/pypi-support/issues/423) so I went for the shorter `sawyer` module name instead.

To install:
```bash
pip install sawyer
```
or
```bash
pip install git+https://github.com/geyang/gym-sawyer
```

## Usage Example

```python
import gym

env = gym.make("sawyer:PickPlace-v0", width=84, height=84, cam_id=-1)

env.render("notebook")
```

<p align="center">
<img width="300px" src="figures/PickPlace-v0-human.png"/>
</p>

![figures/PickPlace-v0.png](figures/PickPlace-v0.png)

Ge Yang, Bradly Stadie © 2020
