import abc
import numpy as np
import typing
from typing import Tuple, Dict
from enum import IntEnum, Enum
from collections import deque

TypeAction = Dict[str, int]


class Actions(IntEnum):
    NOOP = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class ElementsColors(Enum):
    empty = [255, 255, 255]  # WHITE
    agent = [0, 0, 255]  # BLUE
    target = [255, 0, 0]  # RED
    mobile = [0, 128, 128]  # GREEN


class MapElement(IntEnum):
    empty = 0
    agent = 1
    target = 2
    mobile = 3

    def isEmpty(self):
        return self.value == MapElement.empty.value

    def isAgent(self):
        return self.value == MapElement.agent.value

    def isTarget(self):
        return self.value == MapElement.target.value or self.value == MapElement.mobile

    def isMobile(self):
        return self.value == MapElement.mobile


class BaseElem(abc.ABC):
    def __init__(
        self,
        id_elem: int,
        position: Tuple[int, int],
        mobility: bool,
        controllability: bool,
    ):
        self.id = id_elem
        self.position = position
        self.mobile = mobility
        self.controllable = controllability

        if self.isControllable:
            assert self.isMobile

    @property
    def isMobile(self):
        return self.mobile

    @property
    def isControllable(self):
        return self.controllable


class Agent(BaseElem):
    def __init__(self, id_elem: int, position: Tuple[int, int]):
        super().__init__(
            id_elem=id_elem, position=position, mobility=True, controllability=True
        )


class Target(BaseElem):
    def __init__(self, position: Tuple[int, int]):
        super().__init__(
            id_elem=0, position=position, mobility=False, controllability=False
        )
        self.isAlive = True


class MobileTarget(Target):
    def __init__(self, position: Tuple[int, int]):
        super(MobileTarget, self).__init__(position)
        self.mobile = True


class World:

    """
    Implementation of the team catcher game. The interface will collect observations from that game.
    """

    def __init__(
        self, size: int, nb_agents: int, nb_targets: int, nb_mobiles: int, seed: int
    ):
        """

        Parameters
        ----------
        size : int
            Size of the grid (also called map).
        nb_agents : int
            Number of agents.
        nb_targets : int
            Number of targets.
        nb_mobiles : int
            Number of mobile targets.
        seed : int
            Random seed.

        """

        self.seed = seed
        self.size = size
        self.nb_agents = nb_agents
        self.nb_targets = nb_targets
        self.nb_mobiles = nb_mobiles
        self.map = np.zeros((self.size, self.size))  # initialize map
        self.agents = dict()
        self.targets = deque()
        self.mobiles = deque()
        self.capturedTargets = 0
        self.capturedMobiles = 0

    @property
    def nb_targets_alive(self) -> int:
        """
        Total number of targets alive
        Returns
        -------
        int
            Targets alive.
        """
        return len(self.targets) + len(self.mobiles)

    @property
    def totalCaptured(self) -> int:
        """
        Total captured targets after last update
        Returns:
            int
                Total captured targets after last update

        """
        return self.capturedMobiles + self.capturedTargets

    def reset(self):
        # Restart the game
        # At each episode the targets are resuscitated.

        all_positions = [
            (i, j) for j in range(1, self.size - 1) for i in range(1, self.size - 1)
        ]
        random_idx = np.arange(0, (self.size - 2) * (self.size - 2))
        np.random.shuffle(random_idx)

        agents_pos = [all_positions[random_idx[i]] for i in range(self.nb_agents)]
        # first nb_agents are randomly selected

        targets_pos = [
            all_positions[random_idx[i]]
            for i in range(
                self.nb_agents, self.nb_agents + self.nb_targets + self.nb_mobiles
            )
        ]
        # same for targets
        self.agents = {
            f"agent_{i+1}": Agent(id_elem=(i + 1), position=agents_pos[i])
            for (i, pos) in enumerate(agents_pos)
        }

        for i in range(self.nb_targets + self.nb_mobiles):
            if i < self.nb_targets:
                self.targets.append(Target(position=targets_pos[i]))
            else:
                self.mobiles.append(MobileTarget(position=targets_pos[i]))

        self._fill_map()
        self._do_captures()
        self._update_position_state()

    def _fill_map(self):
        """
        Fill map with agents and targets that are still alive
        Returns:

        """
        # start with empty cells
        self.map = np.ones((self.size, self.size)) * MapElement.empty

        # add agents
        for _, agent in self.agents.items():
            agentPosition = agent.position
            self.map[agentPosition[0], agentPosition[1]] = MapElement.agent
        # add targets
        for target in self.targets + self.mobiles:
            targetPosition = target.position
            if target.isAlive:
                if target.isMobile:
                    self.map[targetPosition[0], targetPosition[1]] = MapElement.mobile
                else:
                    self.map[targetPosition[0], targetPosition[1]] = MapElement.target
            else:
                # remove dead targets
                if target.isMobile:
                    self.mobiles.remove(target)
                else:
                    self.targets.remove(target)

    @property
    def get_state(self) -> dict:
        return self.state

    @property
    def state(self) -> dict:
        return {"map": self.map, "agent_position": self.agent_position}

    def _update_position_state(self):
        # Updates the current state of the game (which will be returned by the step and reset method of the gym interface)
        self.agent_position = {k: v.position for k, v in self.agents.items()}

    def _update_agent(self, action: int, agent_id: str):
        """
        Move an agent on the map
        Parameters
        ----------
        action : int
            Action in {0,1,2,3,4}.
        agent_id : str
            Name of the agent : "agent_1" ..., "agent_n"

        """
        agent = self.agents[agent_id]
        self._update_mobile(action, agent, True)

    def _update_mobile(self, action: int, mobile: BaseElem, isAgent: bool):
        """
        Move a mobile element (agent or mobile target) on the map

        Parameters
        ----------
        action : int
            Action type.
        mobile : BaseElem
        isAgent : bool
            Indicate if the mobile is an agent (True) or a target (False)

        """

        if isAgent:
            element = MapElement.agent
        else:
            element = MapElement.target

        # We check if an action is feasible and do it.
        if action == Actions.UP:
            if (
                mobile.position[0] - 1 > 0
                and self.map[mobile.position[0] - 1, mobile.position[1]]
                == MapElement.empty
            ):
                self.map[mobile.position[0] - 1, mobile.position[1]] = element
                self.map[mobile.position[0], mobile.position[1]] = MapElement.empty
                mobile.position = (mobile.position[0] - 1, mobile.position[1])

        if action == Actions.DOWN:
            if (
                mobile.position[0] + 1 < self.size
                and self.map[mobile.position[0] + 1, mobile.position[1]]
                == MapElement.empty
            ):
                self.map[mobile.position[0] + 1, mobile.position[1]] = element
                self.map[mobile.position[0], mobile.position[1]] = MapElement.empty
                mobile.position = (mobile.position[0] + 1, mobile.position[1])

        if action == Actions.LEFT:
            if (
                mobile.position[1] - 1 > 0
                and self.map[mobile.position[0], mobile.position[1] - 1]
                == MapElement.empty
            ):
                self.map[mobile.position[0], mobile.position[1] - 1] = element
                self.map[mobile.position[0], mobile.position[1]] = MapElement.empty
                mobile.position = (mobile.position[0], mobile.position[1] - 1)

        if action == Actions.RIGHT:
            if (
                mobile.position[1] + 1 < self.size
                and self.map[mobile.position[0], mobile.position[1] + 1]
                == MapElement.empty
            ):
                self.map[mobile.position[0], mobile.position[1] + 1] = element
                self.map[mobile.position[0], mobile.position[1]] = MapElement.empty
                mobile.position = (mobile.position[0], mobile.position[1] + 1)

    @classmethod
    def agent_capture(
        cls, pixel: Tuple[int, int], size: int, world_map: np.ndarray
    ) -> bool:
        """Check if a target is captured by the agents

        Parameters
        ----------
        pixel : Tuple[int,int]
            Target location.
        size : int
            Size of the map.
        world_map : np.ndarray
            The map.

        Returns
        -------
        bool
            True if a target is captured, else False.

        """
        potential_neighborhood = [
            (pixel[0] + 1, pixel[1]),
            (pixel[0] - 1, pixel[1]),
            (pixel[0], pixel[1] + 1),
            (pixel[0], pixel[1] - 1),
        ]
        n_agent_neighbour = 0

        for neighbour in potential_neighborhood:
            if 0 <= neighbour[0] < size and 0 <= neighbour[1] < size:
                if world_map[neighbour[0], neighbour[1]] == MapElement.agent:
                    n_agent_neighbour += 1
        return n_agent_neighbour >= 2

    def _do_captures(self) -> Tuple[int, int]:
        """For each target, we check if it is captured and update its status.
        The number of captures for each target type is returned."""

        nTargetCaptures = 0
        nMobileCaptures = 0

        for target in self.targets + self.mobiles:
            targetPosition = target.position
            if self.agent_capture(
                (targetPosition[0], targetPosition[1]), self.size, self.map
            ):
                target.isAlive = False
                if target.isMobile:
                    nMobileCaptures += 1
                else:
                    nTargetCaptures += 1

        return nTargetCaptures, nMobileCaptures

    def _move_mobiles(self):
        for mobile in self.mobiles:
            if mobile.isAlive:
                # select a random action
                action = Actions(np.random.randint(len(Actions)))
                self._update_mobile(action, mobile, False)

    def update(self, joint_action: Actions):
        """Update map, agents and targets state

        Parameters
        ----------
        joint_action : TypeAction
            Dict of actions for each agent : {"agent_1" : 0, "agent_2" : 3, ..., "agent_n": 1}
        """

        for agent_id, action in joint_action.items():
            self._update_agent(action=action, agent_id=agent_id)
        self._move_mobiles()
        self.capturedTargets, self.capturedMobiles = self._do_captures()
        self._fill_map()
        self._update_position_state()
