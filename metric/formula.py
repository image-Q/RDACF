# -*- coding: utf-8 -*-


import numpy as np
from .types import Costs


def costs_to_score(costs: Costs):

    dist_to_destination = costs.dist_to_destination
    time = costs.steps

    humanness_error = np.mean([
        costs.dist_to_obstacles,
        costs.jerk_linear,
        costs.lane_center_offset
    ])

    rule_violation = np.mean([
        costs.speed_limit,
        costs.wrong_way
    ])

    overall = (
        0.25 * (1 - dist_to_destination)
        + 0.25 * (1 - time)
        + 0.25 * (1 - humanness_error)
        + 0.25 * (1 - rule_violation)
    )

    return {
        "overall": overall,
        "dist_to_destination": dist_to_destination,
        "time": time,
        "humanness_error": humanness_error,
        "rule_violation": rule_violation,
    }
