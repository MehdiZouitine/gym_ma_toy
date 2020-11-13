from gym import error, spaces, utils
import numpy as np


def create_observation_space(
    grid_size: int, nb_agents: int, partially_observable: bool
):
    if partially_observable:
        return spaces.Dict(
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
                "partial_map": spaces.Box(
                    low=0, high=2, shape=(grid_size, grid_size), dtype=np.float32
                ),
            }
        )
    return spaces.Dict(
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
