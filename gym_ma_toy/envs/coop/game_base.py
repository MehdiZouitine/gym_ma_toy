import abc
import numpy as np

from typing import Tuple, Dict
from aenum import IntEnum, Enum
from collections import deque

TypeAction = Dict[str, int]


class Actions(IntEnum):
    NOOP = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    UP_RIGHT = 5
    UP_LEFT = 6
    DOWN_RIGHT = 7
    DOWN_LEFT = 8


class ElementsColors(Enum):
    agent_diag = [0, 128, 128]  # BLUE
    agent_hv = [0, 128, 255]  # BLUE
    agent = [0, 0, 255]  # BLUE
    empty = [255, 255, 255]  # WHITE
    mobile = [255, 0, 0]  # RED
    target = [255, 128, 0]  # ORANGE


class MapElement(IntEnum):
    agent_diag = -3
    agent_hv = -2
    agent = -1
    empty = 0
    mobile = 1
    target = 2

    def isEmpty(self):
        return self.value == MapElement.empty.value

    def isAgent(self):
        return self.value < MapElement.empty.value

    def isTarget(self):
        return self.value > MapElement.empty.value

    def isMobile(self):
        return self.value == MapElement.mobile


class BaseElem(abc.ABC):
    def __init__(
        self,
        id_elem: int,
        position: Tuple[int, int],
        mobility: bool,
        controllability: bool,
        has_hv: bool,
        has_diag: bool,
    ):
        self.id = id_elem
        self.position = position
        self.mobile = mobility
        self.controllable = controllability
        self.hasHV = has_hv
        self.hasDiag = has_diag
        if self.isControllable:
            assert self.isMobile

    @property
    def isMobile(self):
        return self.mobile

    @property
    def isControllable(self):
        return self.controllable

    @property
    def isOmniMobile(self):
        return self.isMobile and self.hasHV and self.hasDiag


class Agent(BaseElem):
    def __init__(
        self,
        id_elem: int,
        position: Tuple[int, int],
        has_hv: bool = True,
        has_diag: bool = False,
    ):
        super().__init__(
            id_elem=id_elem,
            position=position,
            mobility=True,
            controllability=True,
            has_hv=has_hv,
            has_diag=has_diag,
        )


class Target(BaseElem):
    def __init__(
        self, position: Tuple[int, int], has_hv: bool = False, has_diag: bool = False
    ):
        super().__init__(
            id_elem=0,
            position=position,
            mobility=False,
            controllability=False,
            has_hv=has_hv,
            has_diag=has_diag,
        )
        self.isAlive = True


class MobileTarget(Target):
    def __init__(self, position: Tuple[int, int]):
        super(MobileTarget, self).__init__(position, has_hv=True, has_diag=False)
        self.mobile = True


class WorldBase:

    """
    Implementation of the team catcher game. The interface will collect observations from that game.
    """

    def __init__(
        self,
        size: int,
        nb_agents_hv: int,
        nb_agents_diag: int,
        nb_targets: int,
        nb_mobiles: int,
        seed: int,
    ):
        """

        Parameters
        ----------
        size : int
            Size of the grid (also called map).
        nb_agents_hv : int
            Number of agents with horizontal/vertical mobility.
        nb_agents_diag : int
            Number of agents with horizontal/vertical mobility.
        nb_targets : int
            Number of targets.
        nb_mobiles : int
            Number of mobile targets.
        seed : int
            Random seed.

        """

        self.seed = seed
        self.size = size
        self.nb_agents_hv = nb_agents_hv
        self.nb_agents_diag = nb_agents_diag
        nb_agents = nb_agents_hv + nb_agents_diag
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

        targets_pos = [
            all_positions[random_idx[i]]
            for i in range(
                self.nb_agents, self.nb_agents + self.nb_targets + self.nb_mobiles
            )
        ]

        n_hv = self.nb_agents_hv
        n_diag = self.nb_agents_diag
        for (agentIdx, pos) in enumerate(agents_pos):
            if n_hv > 0:
                self.agents.update(
                    {
                        f"agent_{agentIdx + 1}": Agent(
                            id_elem=(agentIdx + 1),
                            position=agents_pos[agentIdx],
                            has_hv=True,
                            has_diag=False,
                        )
                    }
                )
                n_hv -= 1
            elif n_diag > 0:
                self.agents.update(
                    {
                        f"agent_{agentIdx + 1}": Agent(
                            id_elem=(agentIdx + 1),
                            position=agents_pos[agentIdx],
                            has_hv=False,
                            has_diag=True,
                        )
                    }
                )
                n_diag -= 1

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
            element = MapElement.agent
            if not agent.isOmniMobile:
                if not agent.hasHV:
                    element = MapElement.agent_diag
                else:
                    element = MapElement.agent_hv
            self.map[agentPosition[0], agentPosition[1]] = element
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

        if (
            action != Actions.NOOP
            and (mobile.hasHV or mobile.hasDiag)
            and (
                (mobile.hasHV and action <= Actions.RIGHT)
                or (mobile.hasDiag and action >= Actions.UP_RIGHT)
            )
        ):
            if isAgent:
                element = MapElement.agent
            else:
                element = MapElement.target

            # list of possible actions
            possibleActions = [False] * (len(Actions) - 1)  # up,down,
            # left,right, up_right, up_left, down_right, down_left
            upDownLeftRight = [Actions.UP, Actions.DOWN, Actions.LEFT, Actions.RIGHT]
            movements = [
                [-1, 0, 0],
                [1, 0, self.size],
                [0, -1, 0],
                [0, 1, self.size],
                # no need for boundaries for diagonal movements
                [-1, 1],
                [-1, -1],
                [1, 1],
                [1, -1],
            ]
            for actionIdx in range(len(upDownLeftRight)):
                shift0 = movements[actionIdx][0]
                shift1 = movements[actionIdx][1]
                boundary = movements[actionIdx][2]
                if actionIdx < 2:  # test UP and DOWN
                    if boundary == 0:
                        possibleActions[actionIdx] = (
                            mobile.position[0] + shift0 > boundary
                            and self.map[
                                mobile.position[0] + shift0, mobile.position[1]
                            ]
                            == MapElement.empty
                        )
                    else:
                        possibleActions[actionIdx] = (
                            mobile.position[0] + shift0 < boundary
                            and self.map[
                                mobile.position[0] + shift0, mobile.position[1]
                            ]
                            == MapElement.empty
                        )
                else:  # test LEFT and RIGHT
                    if boundary == 0:
                        possibleActions[actionIdx] = (
                            mobile.position[1] + shift1 > boundary
                            and self.map[
                                mobile.position[0], mobile.position[1] + shift1
                            ]
                            == MapElement.empty
                        )
                    else:
                        possibleActions[actionIdx] = (
                            mobile.position[1] + shift1 < boundary
                            and self.map[
                                mobile.position[0], mobile.position[1] + shift1
                            ]
                            == MapElement.empty
                        )

            # fill the remaining possible actions by testing the HV ones
            possibleActions[4] = possibleActions[0] and possibleActions[3]
            possibleActions[5] = possibleActions[0] and possibleActions[2]
            possibleActions[6] = possibleActions[1] and possibleActions[3]
            possibleActions[7] = possibleActions[1] and possibleActions[2]

            # do the action if it is possible
            actionIdx = int(action) - 1  # not considering NOOP
            if possibleActions[actionIdx]:
                shift0 = movements[actionIdx][0]
                shift1 = movements[actionIdx][1]
                # action is possible, let's do it
                self.map[
                    mobile.position[0] + shift0, mobile.position[1] + shift1
                ] = element
                mobile.position = (
                    mobile.position[0] + shift0,
                    mobile.position[1] + shift1,
                )
                self.map[mobile.position[0], mobile.position[1]] = MapElement.empty

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
