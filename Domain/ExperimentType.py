from enum import Enum


class ExperimentType(Enum):
    EPISODES = 0
    CHANGE_ORIGIN = 1
    CHANGE_GOAL = 2
    DISABLE_TILE = 3
    CHANGE_DIMENSION = 4
