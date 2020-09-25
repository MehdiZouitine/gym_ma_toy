from typing import Tuple

import nympy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding


CELL_SIZE = 5

ACTION_MEANING = {0: "NOOP", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}

ELEMENTNS_COLORS = {0: "white", 1: "blue", 2: "red"}  # empty  # Agent  # Target


class Map:
    pass


class _Agent:
    def __init__(self, position: Tuple[int, int]):
        self.position = position
        self.value = 1

    def update_position(self, position: Tuple[int, int]) -> None:
        self.position = position


class TeamCatcher(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, grid_size: int = 64, nb_agent: int = 4):

        self.action_space = spaces.Dict(
            {f"agent_{i+1}": spaces.Discrete(5) for i in range(nb_agent)}
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
                        for i in range(nb_agent)
                    }
                ),
            }
        )
    
action = {'agent_1': 1, "agent_2": 3}

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode="human", close=False):
        pass

    @classmethod
    def map2image(cls, map):
        return image

    @classmethod
    def check_position_agents(cls, agents_position, agents_actions):
        pass

    @classmethod
    def compute_reward(cls, agents_actions, target_positions):
        pass

    @classmethod
    def delete_target(cls, map, target_position_delete):
        pass
