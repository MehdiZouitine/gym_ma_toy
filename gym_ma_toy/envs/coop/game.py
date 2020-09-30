import abc
from typing import Tuple
import numpy as np

ACTIONS = {"NOOP": 0, "UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4}


class BaseElem(abc.ABC):
    def __init__(self, id_elem: int, position: Tuple[int, int]):
        self.id = id_elem
        self.position = position


class Agent(BaseElem):
    def __init__(self, id_elem: int, position: Tuple[int, int]):
        super().__init__(id_elem=id_elem, position=position)


class Target(BaseElem):
    def __init__(self, position):
        self.position = position


class World:
    def __init__(self, size, nb_agents, nb_targets, seed):
        self.seed = seed
        self.size = size
        self.nb_agents = nb_agents
        self.nb_targets = nb_targets
        self.targets_alive = self.nb_targets

    @property
    def nb_targets_alive(self):
        return self.targets_alive

    def reset(self):
        self.targets_alive = self.nb_targets
        self.map = np.zeros((self.size, self.size))
        all_positions = [
            (i, j) for j in range(1, self.size - 1) for i in range(1, self.size - 1)
        ]

        random_idx = np.arange(0, (self.size - 2) * (self.size - 2))
        np.random.shuffle(random_idx)

        agents_pos = [all_positions[random_idx[i]]
                      for i in range(self.nb_agents)]

        targets_pos = [
            all_positions[random_idx[i]]
            for i in range(self.nb_agents, self.nb_agents + self.nb_targets)
        ]

        self.agents = {
            f"agent_{i+1}": Agent(id_elem=(i + 1), position=agents_pos[i])
            for (i, pos) in enumerate(agents_pos)
        }

        self.targets = [Target(position=pos) for pos in targets_pos]

        for pos in agents_pos:
            self.map[pos[0], pos[1]] = 1
        for pos in targets_pos:
            self.map[pos[0], pos[1]] = 2

        self._update_map()
        self._update_position_state()

    @property
    def get_state(self):
        return self.state

    def _update_position_state(self):
        self.agent_position = {k: v.position for k, v in self.agents.items()}
        self.state = {"map": self.map, "agent_position": self.agent_position}

    def _update_action(self, action, agent_id):
        # if action == ACTIONS['NOOP']:
        #     return True
        agent = self.agents[agent_id]
        if action == ACTIONS["UP"]:
            if (
                agent.position[0] - 1 > 0
                and self.map[agent.position[0] - 1, agent.position[1]] == 0
            ):
                self.map[agent.position[0] - 1, agent.position[1]] = 1
                self.map[agent.position[0], agent.position[1]] = 0
                agent.position = (agent.position[0] - 1, agent.position[1])

        if action == ACTIONS["DOWN"]:
            if (
                agent.position[0] + 1 < self.size
                and self.map[agent.position[0] + 1, agent.position[1]] == 0
            ):
                self.map[agent.position[0] + 1, agent.position[1]] = 1
                self.map[agent.position[0], agent.position[1]] = 0
                agent.position = (agent.position[0] + 1, agent.position[1])

        if action == ACTIONS["LEFT"]:
            if (
                agent.position[1] - 1 > 0
                and self.map[agent.position[0], agent.position[1] - 1] == 0
            ):
                self.map[agent.position[0], agent.position[1] - 1] = 1
                self.map[agent.position[0], agent.position[1]] = 0
                agent.position = (agent.position[0], agent.position[1] - 1)

        if action == ACTIONS["RIGHT"]:
            if (
                agent.position[1] + 1 < self.size
                and self.map[agent.position[0], agent.position[1] + 1] == 0
            ):
                self.map[agent.position[0], agent.position[1] + 1] = 1
                self.map[agent.position[0], agent.position[1]] = 0
                agent.position = (agent.position[0], agent.position[1] + 1)

    @classmethod
    def agent_capture(cls, pixel, size, map):
        potential_neighborhood = [
            (pixel[0] + 1, pixel[1]),
            (pixel[0] - 1, pixel[1]),
            (pixel[0], pixel[1] + 1),
            (pixel[0], pixel[1] - 1),
        ]
        n_agent_neighbour = 0

        for neighbour in potential_neighborhood:
            if (
                neighbour[0] >= 0
                and neighbour[0] < size
                and neighbour[1] >= 0
                and neighbour[1] < size
            ):
                if map[neighbour[0], neighbour[1]] == 1:
                    n_agent_neighbour += 1
        return n_agent_neighbour >= 2

    def _update_map(self):
        size = self.size
        for i in range(size):
            for j in range(size):
                if self.map[i, j] == 2:
                    if self.agent_capture((i, j), size, self.map):
                        self.map[i, j] = 0
                        self.targets_alive -= 1

    def update(self, joint_action):

        for agent_id, action in joint_action.items():
            self._update_action(action=action, agent_id=agent_id)
        self._update_map()
        self._update_position_state()
