import abc
import numpy as np
import typing
from typing import Tuple, Dict


TypeAction = Dict[str, int]


ACTIONS = {"NOOP": 0, "UP": 1, "DOWN": 2,
    "LEFT": 3, "RIGHT": 4}  # actions meaning


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

     """
    Implementation of the team catcher game. The interface will collect observations from that game.
    """
    def __init__(self, size:int, nb_agents:int, nb_targets:int, seed:int):
        """

        Parameters
        ----------
        size : int
            Size of the grid (also called map).
        nb_agents : int
            Number of agents.
        nb_targets : int
            Number of targets.
        seed : int
            Random seed.

        """
        self.seed = seed
        self.size = size
        self.nb_agents = nb_agents
        self.nb_targets = nb_targets
        self.targets_alive = self.nb_targets

    @property
    def nb_targets_alive(self) -> int:
        """Short summary.

        Returns
        -------
        int
            Targets alive.

        """
        return self.targets_alive

    def reset(self):
        # Restart the game
        self.targets_alive = self.nb_targets # At each episode the targets are resuscitated.
        self.map = np.zeros((self.size, self.size)) # and the map is cleaned

        '''We want to place 3 targets and 3 targets randomly on the private grid of its border.
         For that we generate the set of couple (i,j) of our private border map.
         Then we randomly select nb_agents element in this list and then nb_targets. '''

        all_positions = [
            (i, j) for j in range(1, self.size - 1) for i in range(1, self.size - 1)
        ]
        random_idx = np.arange(0, (self.size - 2) * (self.size - 2))
        np.random.shuffle(random_idx)

        agents_pos = [all_positions[random_idx[i]] for i in range(self.nb_agents)]
        # first nb_agents are randomly selected

        targets_pos = [
            all_positions[random_idx[i]]
            for i in range(self.nb_agents, self.nb_agents + self.nb_targets)
        ]
        # same for targets
        self.agents = {
            f"agent_{i+1}": Agent(id_elem=(i + 1), position=agents_pos[i])
            for (i, pos) in enumerate(agents_pos)
        }

        self.targets = [Target(position=pos) for pos in targets_pos]

        for pos in agents_pos:
            self.map[pos[0], pos[1]] = 1 #Agents are represented by 1
        for pos in targets_pos:
            self.map[pos[0], pos[1]] = 2 # Targets by 2

        self._update_map()
        self._update_position_state()

    @property
    def get_state(self):
        return self.state

    def _update_position_state(self):
        # Updates the current state of the game (which will be returned by the step and reset method of the gym interface)
        self.agent_position = {k: v.position for k, v in self.agents.items()}
        self.state = {"map": self.map, "agent_position": self.agent_position}

    def _update_action(self, action:int, agent_id:str):
        """

        Parameters
        ----------
        action : int
            Action in {0,1,2,3,4}.
        agent_id : str
            Name of the agent : "agent_1" ..., "agent_n"

        """
        # We check if an action is feasible and do it.
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
    def agent_capture(cls, pixel:Tuple[int,int], size:int, map:np.ndarray)-> bool:
        """ Check if a target is captured by the agents

        Parameters
        ----------
        pixel : Tuple[int,int]
            Target location.
        size : int
            Size of the map.
        map : np.ndarray
            The map.

        Returns
        -------
        bool
            True if an targets is captured, else False.

        """
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
        ''' For each target, we check if it is captured and if so,
         we delete it. So we reduce the number of live targets by 1.'''
        size = self.size
        for i in range(size):
            for j in range(size):
                if self.map[i, j] == 2:
                    if self.agent_capture((i, j), size, self.map):
                        self.map[i, j] = 0
                        self.targets_alive -= 1

    def update(self, joint_action:TypeAction):
        """Update the map and the agents and targets state

        Parameters
        ----------
        joint_action : TypeAction
            Dict of actions for each agent : {"agent_1" : 0, "agent_2" : 3, ..., "agent_n": 1}
        """

        for agent_id, action in joint_action.items():
            self._update_action(action=action, agent_id=agent_id)
        self._update_map()
        self._update_position_state()
