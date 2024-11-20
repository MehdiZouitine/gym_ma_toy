from typing import Tuple, Dict, Union, Any, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .render_utils import render_observable, render_partially_observable
from .space_utils import create_observation_space
from .game_base import WorldBase, Actions

TypeObservation = Dict[str, Union[np.ndarray, Dict[str, int]]]
NB_ACTIONS = len(Actions)


class TeamCatcherBase(gym.Env):
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
        4: RIGHT

    Parameters:
        grid_size (int): The classifier to bag. Defaults to `64`.
        nb_agents (int): The number of agents. Defaults to `256`.
        nb_targets (int): The number of target to catch. Defaults to `128`.
        seed (int): Random number generator seed for reproducibility. Defaults to `None`.

    Example:

        In the following example we will show a classic gym loop
        using the team catcher environment.

        >>> import gymnasium as gym
        >>> import gym_ma_toy

        >>> env = gym.make('team_catcher-v0')
        >>> done = False
        >>> truncated = False
        >>> obs, info = env.reset()
        >>> while not done or not truncated:
        ...    env.render()
        ...    obs, reward, truncated, done, info = env.step(env.action_space.sample())
        >>> env.close()

    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        grid_size: int = 64,
        nb_agents_hv: int = 128,
        nb_agents_diag: int = 128,
        nb_targets: int = 128,
        nb_mobiles: int = 32,
        fow_agents_hv: int = 0,
        fow_agents_diag: int = 0,
        seed: Optional[int] = None,
    ):
        nb_agents = nb_agents_hv + nb_agents_diag
        if (grid_size - 1) ** 2 < nb_agents + nb_targets:
            population = nb_agents + nb_targets + nb_mobiles
            maximum_population = (grid_size - 1) ** 2
            raise ValueError(
                f" nb_agents + nb_targets + nb_mobiles ({population}) should "
                f"be less than (grid_size - 1) ** 2 ({maximum_population})"
            )

        self.grid_size = grid_size
        self.partially_observable = (fow_agents_hv + fow_agents_diag) > 0
        self.action_space = spaces.Box(
            low=0, high=NB_ACTIONS - 1, shape=(grid_size, grid_size), dtype=np.int32
        )
        self.observation_space = create_observation_space(
            grid_size=grid_size,
            nb_agents=nb_agents,
            partially_observable=self.partially_observable,
        )

        self.world = WorldBase(
            size=grid_size,
            nb_agents_hv=nb_agents_hv,
            nb_agents_diag=nb_agents_diag,
            nb_targets=nb_targets,
            nb_mobiles=nb_mobiles,
            fow_agents_hv=fow_agents_hv,
            fow_agents_diag=fow_agents_diag,
            seed=seed,
        )

        self.nb_targets_alive = self.world.nb_targets_alive

        self.obs: TypeObservation = None  # For render
        self.viewer = None  #  For render
        self.grid_size = grid_size  # For render

        self.nb_step: int = None
        self.seed(seed)

    def step(
        self, action: Actions
    ) -> Tuple[TypeObservation, float, bool, Dict[str, Any]]:
        self.world.update(action)  # apply action to the engine

        self.obs = self.world.get_state

        reward = self.compute_reward(
            capturedTargets=self.world.capturedTargets,
            capturedMobiles=self.world.capturedMobiles,
        )
        self.nb_targets_alive = self.world.nb_targets_alive

        done = self.episode_end(current_nb_targets_alive=self.nb_targets_alive)
        self.nb_step += 1
        info = {"step": self.nb_step, "target alive": self.nb_targets_alive}

        return self.obs, reward, done, False, info

    def reset(self, seed=None, options=None) -> TypeObservation:
        self.world.reset()
        self.obs = self.world.get_state
        self.nb_step = 0
        self.nb_targets_alive = self.world.nb_targets_alive
        return self.obs, {"step": self.nb_step, "target alive": self.nb_targets_alive}

    def render(self, close=False, fig_size=8):
        if self.partially_observable:
            image = render_partially_observable(
                grid_size=self.grid_size,
                obs=self.obs,
                fig_size=fig_size,
            )
        else:
            image = render_observable(
                grid_size=self.grid_size,
                obs=self.obs,
                fig_size=fig_size,
            )
        return image

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed: int = None):
        np.random.seed(seed)
        return

    @classmethod
    def compute_reward(cls, capturedTargets: int, capturedMobiles: int) -> int:
        # - 1 at each step to encourage the agents to catch the targets ASAP (survival reward)
        # double points are awarded for captured mobiles
        return -1 + capturedTargets + 2 * capturedMobiles

    @classmethod
    def episode_end(cls, current_nb_targets_alive: int) -> bool:
        # If the number of targets is null then the episode is over.
        if current_nb_targets_alive == 0:
            return True
        return False
