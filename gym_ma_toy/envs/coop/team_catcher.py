from typing import Tuple, Dict, Union, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.classic_control import rendering

from . import game

# TODO IMPLEMENT SEED METHOD

TypeObservation = Dict[str, Union[np.ndarray, Dict[str, int]]]
TypeAction = Dict[str, int]

ACTION_MEANING = {0: "NOOP", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}
NB_ACTIONS = len(ACTION_MEANING)

ELEMENTS_COLORS = {
    0: [255, 255, 255],  # WHITE (empty)
    1: [0, 0, 255],  # BLUE (agent)
    2: [255, 0, 0],  # RED (target)
}


class TeamCatcher(gym.Env):
    """
    Interface gym for the team catcher game.
    This is a map where targets are randomly placed.
    The objective of the agents is that there are at least two agents on an adjacent cell of a target to catch it.
    When the target is caught the environment returns a reward point.
    The episode ends when there is no more target on the map.

    Observation dict:
        map: numpy array describe the current state of the game. (0: empty, 1:agent, 2:target)
        agent_position:
            agent_1: tuple of his position (x,y)
            ...
            agent_n: tuple of his positon (x,y)

    Action for an agent:
        0: NOOP
        1: UP
        2: DOWN
        3: LEFT
        4:  RIGHT

    Parameters:
        grid_size (int): The classifier to bag. Defaults to `64`.
        nb_agents (int): The number of agents. Defaults to `256`.
        nb_targets (int): The number of target to catch. Defaults to `128`.
        seed (int): Random number generator seed for reproducibility. Defaults to `None`.

    Example:

        In the following example we will show a classic gym loop
        using the team catcher environment.

        >>> import gym
        >>> import gym_ma_toy

        >>> env = gym.make('team_catcher-v0')
        >>> done = False
        >>> obs = env.reset()
        >>> while not done:
        ...    env.render()
        ...    obs, reward, done, info = env.step(env.action_space.sample())
        >>> env.close()

    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        grid_size: int = 64,
        nb_agents: int = 256,
        nb_targets: int = 128,
        seed: Optional[int] = None,
    ):

        if (grid_size - 1) ** 2 < nb_agents + nb_targets:
            population = nb_agents + nb_targets
            maximum_population = (grid_size - 1) ** 2
            raise ValueError(
                f" nb_agents + nb_targets ({population}) should be less than (grid_size - 1) ** 2 ({maximum_population})"
            )

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

        self.world = game.World(
            size=grid_size, nb_agents=nb_agents, nb_targets=nb_targets, seed=seed
        )

        self.nb_targets_alive = self.world.nb_targets

        self.obs: TypeObservation = None  # For render
        self.viewer = None  #  For render
        self.grid_size = grid_size  # For render

        self.nb_step: int = None

    def step(
        self, action: TypeAction
    ) -> Tuple[TypeObservation, float, bool, Dict[str, Any]]:

        self.world.update(action)  # apply action to the engine

        self.obs = self.world.get_state

        new_nb_agents_alive = self.world.nb_targets_alive
        reward = self.compute_reward(
            current_nb_agents_alive=self.nb_targets_alive,
            new_nb_agents_alive=new_nb_agents_alive,
        )
        self.nb_targets_alive = new_nb_agents_alive

        done = self.episode_end(current_nb_targets_alive=self.nb_targets_alive)
        self.nb_step += 1
        info = {"step": self.nb_step, "target alive": self.nb_targets_alive}

        return self.obs, reward, done, info

    def reset(self) -> TypeObservation:
        self.world.reset()
        self.obs = self.world.get_state
        self.nb_step = 0
        return self.obs

    def render(self, mode="human", close=False, fig_size=8):

        image = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        image[self.obs["map"] == 0] = ELEMENTS_COLORS[0]
        image[self.obs["map"] == 1] = ELEMENTS_COLORS[1]
        image[self.obs["map"] == 2] = ELEMENTS_COLORS[2]

        image = Image.fromarray(image)
        image = image.resize(
            (self.grid_size * fig_size, self.grid_size * fig_size), Image.ANTIALIAS
        )
        image = np.array(image, dtype=np.uint8)
        if mode == "rgb_array":
            return image
        else:
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(image)
            return self.viewer.isopen

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed: int):
        raise NotImplementedError

    @classmethod
    def compute_reward(
        cls, current_nb_agents_alive: int, new_nb_agents_alive: int
    ) -> int:
        # Returns the number of targets caught at time t
        return new_nb_agents_alive - current_nb_agents_alive

    @classmethod
    def episode_end(cls, current_nb_targets_alive: int) -> bool:
        # If the number of targets is null then the episode is over.
        if current_nb_targets_alive == 0:
            return True
        return False
