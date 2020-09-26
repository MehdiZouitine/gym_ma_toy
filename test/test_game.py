from gym_ma_toy.envs.game import World
import numpy as np


class TestGame:
    def test_attribute(self):
        size_list = [10, 20, 100, 200, 1000, 2000, 3000]
        nb_agents_list = [2, 4, 10, 20, 100, 200, 1000]
        nb_targets_list = [2, 4, 10, 20, 100, 200, 1000]
        worlds = [
            World(
                size=size_list[i],
                nb_agents=nb_agents_list[i],
                nb_targets=nb_targets_list[i],
                seed=7,
            )
            for i in range(len(size_list))
        ]
        for i in range(len(size_list)):
            assert (
                (worlds[i].size == size_list[i])
                and (worlds[i].nb_agents == nb_agents_list[i])
                and (worlds[i].nb_targets == nb_targets_list[i])
                and (worlds[i].seed == 7)
            )

    def test_reset(self):
        size_list = [10, 20, 100, 200, 1000, 2000, 3000]
        nb_agents_list = [2, 4, 10, 20, 100, 200, 1000]
        nb_targets_list = [2, 4, 10, 20, 100, 200, 1000]
        worlds = [
            World(
                size=size_list[i],
                nb_agents=nb_agents_list[i],
                nb_targets=nb_targets_list[i],
                seed=7,
            )
            for i in range(len(size_list))
        ]
        for i in range(len(worlds)):
            worlds[i].reset()
            assert (
                (worlds[i].size == size_list[i])
                and (worlds[i].nb_agents == nb_agents_list[i])
                and (worlds[i].nb_targets == nb_targets_list[i])
                and (worlds[i].seed == 7)
                and
                # check if the border are equal to 0
                (np.sum(worlds[i].map[0, :]) == 0)
                and (np.sum(worlds[i].map[:, 0]) == 0)
                and (np.sum(worlds[i].map[worlds[i].size - 1, :]) == 0)
                and (np.sum(worlds[i].map[:, worlds[i].size - 1]) == 0)
                and (np.sum(worlds[i].map) >= worlds[i].nb_agents)
            )
