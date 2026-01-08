# -*- coding: utf-8 -*-

from dataclasses import dataclass

@dataclass
class Costs:
    dist_to_destination: float = 0.0
    steps: float = 0.0
    dist_to_obstacles: float = 0.0
    jerk_linear: float = 0.0
    lane_center_offset: float = 0.0
    speed_limit: float = 0.0
    wrong_way: float = 0.0
    collisions: float = 0.0
    off_road: float = 0.0

@dataclass
class Counts:
    episodes: int = 0
    steps: int = 0
    goals: int = 0

@dataclass
class Metadata:
    difficulty: float = 1.0

@dataclass
class Record:
    costs: Costs = Costs()
    counts: Counts = Counts()
    metadata: Metadata = Metadata()
