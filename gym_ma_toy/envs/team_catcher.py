from typing import Tuple

import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_ma_toy.envs.game import World

ACTION_MEANING = {0: "NOOP", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}
NB_ACTIONS = len(ACTION_MEANING)

ELEMENTS_COLORS = {
    0: [255, 255, 255],  # WHITE (empty)
    1: [0, 0, 255],  # BLUE (agent)
    2: [255, 0, 0],  # RED (target)
}


class TeamCatcher(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, grid_size: int = 64, nb_agents: int = 4, nb_targets: int = 8):
        self.grid_size = grid_size
        self.action_space = spaces.Dict(
            {f"agent_{i+1}": spaces.Discrete(NB_ACTIONS) for i in range(nb_agents)}
        )
        self.observation_space = spaces.Dict(
            {
                "map": spaces.Box(
                    low=0, high=2, shape=(grid_size, grid_size), dtype=np.float32
                ),
                "agent_position": spaces.Dict(
                    {
                        f"agent_{i+1}": spaces.Tuple(
                            (
                                spaces.Discrete(grid_size - 1),
                                spaces.Discrete(grid_size - 1),
                            )
                        )
                        for i in range(nb_agents)
                    }
                ),
            }
        )

        self.world = World(
            grid_size=grid_size, nb_agents=nb_agents, nb_targets=nb_targets
        )

        self.nb_targets_alive = self.world.targets_alive
        self.obs = None  # For render
        self.nb_step = None

    def step(self, action):
        self.world.update(action)

        self.obs = self.world.state

        new_nb_agents_alive = self.world.targets_alive
        reward = self.compute_reward(
            current_nb_agents_alive=self.nb_targets_alive,
            new_nb_agents_alive=new_nb_agents_alive,
        )
        self.nb_targets_alive = new_nb_agents_alive

        done = self.episode_end(current_nb_agents_alive=self.nb_targets_alive)
        info = {"step": self.nb_step, "target alive": self.nb_targets_alive}

        return self.obs, reward, done, info

    def reset(self):
        self.world.reset()
        self.obs = self.world.state
        self.nb_step += 1
        return self.obs

    def render(self, mode="human", close=False):
        image = np.zeros((self.grid_size, self.grid_size, 3))
        image[self.obs == 1] = ELEMENTS_COLORS[1]
        image[self.obs == 2] = ELEMENTS_COLORS[2]

    def seed(self, seed: int):
        pass

    @classmethod
    def compute_reward(
        cls, current_nb_agents_alive: int, new_nb_agents_alive: int
    ) -> int:
        return new_nb_agents_alive - current_nb_agents_alive

    @classmethod
    def episode_end(cls, current_nb_agents_alive: int):
        if current_nb_agents_alive == 0:
            return True
        return False
