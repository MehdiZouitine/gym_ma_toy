## Toy Multi agent environment

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
import gym_ma_toy
import gym
env = env = gym.make('team_catcher-v0')
obs = env.reset()
done = False

while not done:
    obs, reward, done, info = env.step(env.action_space.sample())
    env.render()
```

### Team catcher:

This is a map where targets are randomly placed.
The objective of the agents is that there are at least two agents on an adjacent cell of a target to catch it.
When the target is caught the environment returns a reward point.
The episode ends when there is no more target on the map.



![Alt Text](./img/team_catcher_gif.gif)


The number of agent and target can be arbitrarily large.


![Alt Text](./img/team_catcher_gif_big.gif)



Cite the environment as:
```
@misc{amarl2020
 Author = {Mehdi Zouitine, Adil Zouitine},
 Title = {Toy environment set for multi-agent reinforcement learning and more},
 Year = {2020},
}
```
#### License

This project is free and open-source software licensed under the MIT license.