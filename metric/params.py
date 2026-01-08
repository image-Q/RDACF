# -*- coding: utf-8 -*-

from dataclasses import dataclass

@dataclass
class ParamBase:
    active: bool = True

@dataclass
class Params:
    collisions: ParamBase = ParamBase()
    dist_to_destination: ParamBase = ParamBase()
    jerk_linear: ParamBase = ParamBase()
    lane_center_offset: ParamBase = ParamBase()
    speed_limit: ParamBase = ParamBase()
    steps: ParamBase = ParamBase()
    wrong_way: ParamBase = ParamBase()
    off_road: ParamBase = ParamBase()
