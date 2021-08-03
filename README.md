## Toy Multi agent environment (README IN PROGRESS)

<p align="center">
  <img height="220px" src="https://github.com/MehdiZouitine/gym_ma_toy/blob/master/img/logo.png?raw=true" alt="ma_gym_logo">
</p>
Start of a toy gym environment bank for multi-agent reinforcement learning.

This repo contains for the moment a first environment:

### How to install ?
```
git clone https://github.com/MehdiZouitine/gym_ma_toy
cd gym_ma_toy
pip install -e .
```


### How to use it ?

```python
import gym
import gym_ma_toy

env = gym.make('team_catcher-v0')

obs = env.reset()
done = False
while not done:
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
env.close()
```

### Team catcher:

This is a map where targets are randomly placed. Some targets do not move
, other move randomly (one square at the time).
Agents must capture all targets. Targets are captures when there are at least
 two agents on an adjacent cell of a target.
The environement gives a reward point for each captured target.
The episode ends when there is no more target on the map.
The number of agents and targets can be arbitrarily large.

<p align="center">
<img  src="https://github.com/MehdiZouitine/gym_ma_toy/blob/master/img/big_gif_tc.gif?raw=true" alt="ma_gym_logo">
</p>
There are currently 3 implemented versions:

- **V0**: This is the simplest environment. Targets (ORANGE) do not move and
 agents (BLUE) can only move horizontally and vertically.
 
<p align="center">
<img  src="https://github.com/MehdiZouitine/gym_ma_toy/blob/master/img/squad1.gif?raw=true" alt="ma_gym_logo">
</p>
 <p align="center">
<img  src="https://github.com/MehdiZouitine/gym_ma_toy/blob/master/img/duo_agent?raw=true" alt="ma_gym_logo">
</p>
- **V1**: Some targets do not move (ORANGE) but some can move randomly (RED).
-  <p align="center">
<img  src="https://github.com/MehdiZouitine/gym_ma_toy/blob/master/img/V1.gif?raw=true" alt="ma_gym_logo">
</p>
- **V2**: Some agents move horizontally/vertically (BLUE), other move
 diagonally (GREEN)
- **V3**:  **Partially observable** (fog of war like) environment with some agents move horizontally/vertically (BLUE), other move
 diagonally (GREEN)
#### Some results: 
Training of different team catcher configurations using a Grid-net architecture ([link](http://proceedings.mlr.press/v97/han19a/han19a.pdf)) : 
* 2 agents and 20 targets : 
<p align="center">
<img  src="https://github.com/MehdiZouitine/gym_ma_toy/blob/master/img/sparse_env2agent2.gif?raw=true" alt="ma_gym_logo">
</p>

* 24 agents and 10 targets : 
<p align="center">
<img  src="https://github.com/MehdiZouitine/gym_ma_toy/blob/master/img/150100_train_step_25_step.gif?raw=true" alt="ma_gym_logo">
</p>


### Running multiple environment in parallel

```py
# Running 8 environment in parallel
import gym
import gym_ma_toy

env = gym.vector.make('team_catcher-v0',num_envs=8, asynchronous=True)  
```

### Test

```
pytest test/
```

Cite the environment as:
```
@misc{amarl2020
 Author = {Mehdi Zouitine, Adil Zouitine, Ahmad Berjaoui},
 Title = {Toy environment set for multi-agent reinforcement learning and more},
 Year = {2020},
}
```
#### License

This project is free and open-source software licensed under the MIT license.
