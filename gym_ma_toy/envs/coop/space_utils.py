from gym import error, spaces, utils
import numpy as np


def create_observation_space(
    grid_size: int, nb_agents: int, partially_observable: bool
):
    if partially_observable:
        return spaces.Dict(
            {
                "map": spaces.Box(
                    low=-3, high=2, shape=(grid_size, grid_size), dtype=np.float32
                ),
                "partial_map": spaces.Box(
                    low=-3, high=2, shape=(grid_size, grid_size), dtype=np.float32
                ),
                "agent_mask": spaces.Box(
                    low=1, high=1, shape=(grid_size, grid_size), dtype=np.int32
                ),
            }
        )
    return spaces.Dict(
        {
            "map": spaces.Box(
                low=-3, high=2, shape=(grid_size, grid_size), dtype=np.float32
            ),
            "agent_mask": spaces.Box(
                    low=1, high=1, shape=(grid_size, grid_size), dtype=np.int32
                ),
        }
    )
