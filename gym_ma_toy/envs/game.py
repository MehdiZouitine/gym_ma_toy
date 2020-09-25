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

    @property
    def name(self):
        return f"agent_{self.id}"


class Target(BaseElem):
    def __init__(self, position):
        self.position = position
        self.is_alive = True


class World:
    def __init__(self, grid_size, nb_agents, nb_targets):

        self.size = size
        self.nb_agents = nb_agents
        self.nb_targets = nb_targets

    @ property
    def nb_targets_alive(self):
        return self.nb_targets

    def reset(self):
        all_positions = [(i, j) for j in range(self.size)
                          for i in range(self.size)]
        random_idx = np.random.shuffle(np.arange(self.size * self.size))

        agents_pos = [all_positions[random_idx[i]]
            for i in range(self.nb_agent)]

        targets_pos = [all_positions[random_idx[i]]
            for i in range(self.nb_agent, self.nb_agent + self.targets)]

        self.agents = {
            f"agent_{i+1}": Agent(id_elem=(i + 1), position=agent_pos[i] for i, pos in enumerate(agents_pos)}

        self.targets=[Target(position=all_positions[random_idx[i]) for pos in targets_pos]


        self.map=np.zeros((self.size, self.size))

        for pos in agents_pos:
            map[pos[0], pos[1]]=1
        for pos in targets_pos:
            map[pos[0], pos[1]]=2

    @ classmethod
    def update_action(cls, action, agent, size, map):
        if action == ACTIONS['NOOP']:
            return True

        if action == ACTIONS['UP']:
            if agent.position[0] + -1 > 0 and map[agent.position[0] - 1, agent.position[1]] == 0:
                agent.position=(agent.position[0] - 1, agent.position[1])

        if action == ACTIONS['DOWN']:
            if agent.position[0] + 1 < size and map[agent.position[0] + 1, agent.position[1]] == 0:
                agent.position=(agent.position[0] + 1, agent.position[1])

        if action == ACTIONS['LEFT']:
            if agent.position[1] > 0 and map[agent.position[0], agent.position[1] - 1] == 0:
                agent.position=(agent.position[0], agent.position[1] - 1)

        if action == ACTIONS['RIGHT']:
            if agent.position[1] < size and map[agent.position[0], agent.position[1] + 1] == 0:
                agent.position=(agent.position[0], agent.position[1] + 1)

    @ classmethod
    def update_map(map):

    def moove(action):
        if action

    def update(self, joint_action):

        for agent, action in joint_action:
            self.update_action(action=action, agent=agent,
                        size=self.size, map=self.map)
