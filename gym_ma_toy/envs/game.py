import abc
from typing import Tuple


class BaseElem(abc.ABC):
    def __init__(self, id_elem: int, position: Tuple[int, int]):
        self.id = id_elem
        self.position = position

    @property
    @abc.abstractmethod
    def name(self):
        pass


class Agent:
    def __init__(self, id_elem: int, position: Tuple[int, int]):
        self.id = id_elem
        self.position = position

    def update_position(self):
        pass

    @property
    def name(self):
        return f"agent_{self.id}"


class Target(BaseElem):
    @property
    def name(self):
        pass
