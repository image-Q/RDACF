# -*- coding: utf-8 -*-

import math
import numpy as np
from .types import Costs
from .utils import running_mean, safe_div


class CostFuncsV1:

    def __init__(self):
        self.reset()

    # --------------------------------------------------------------
    def reset(self):

        self.prev_v = None
        self.prev_a = None
        self.jerk_mean = 0.0
        self.jerk_step = 0
        self.lane_off_sum = 0.0
        self.speed_over_sum = 0.0
        self.dist_to_obs_vals = []       # 邻车距离 cost
        self.steps = 0
        self.wrong = False
        self.collided = False
        self.offroad = False
        self.dist_goal = None
        self.start_dist = None
        self.start_pos = None
        self.goal = None

    # --------------------------------------------------------------
    def step(self, o, dt=0.1):

        ego_pos = np.array(o["ego_state"][-1, :2])
        ego_heading = o["ego_state"][-1, 2]
        ego_speed = o["ego_state"][-1, 3]

        if self.start_pos is None:
            self.start_pos = ego_pos
            self.goal = np.array(o["goal_pos"]) if "goal_pos" in o else np.array([0, 0])
            self.start_dist = np.linalg.norm(self.goal - ego_pos)
        self.dist_goal = np.linalg.norm(self.goal - ego_pos)

        vx, vy = ego_speed * math.cos(ego_heading), ego_speed * math.sin(ego_heading)
        if self.prev_v is not None:
            ax, ay = (vx - self.prev_v[0]) / dt, (vy - self.prev_v[1]) / dt
            if self.prev_a is not None:
                jx, jy = (ax - self.prev_a[0]) / dt, (ay - self.prev_a[1]) / dt
                jerk = math.hypot(jx, jy)
                self.jerk_mean, self.jerk_step = running_mean(self.jerk_mean, self.jerk_step, jerk)
            self.prev_a = (ax, ay)
        self.prev_v = (vx, vy)

        ev = o.get("events", None)
        if ev:
            self.wrong |= bool(getattr(ev, "wrong_way", False))
            self.collided |= len(getattr(ev, "collisions", [])) > 0
            self.offroad |= bool(getattr(ev, "off_road", False))


        lane_off = float(o.get("lane_center_offset", 0.0))
        self.lane_off_sum += (lane_off) ** 2

        lim = float(o.get("speed_limit", 13.9))
        over = max(ego_speed - lim, 0.0)
        over_norm = min((over / (0.5 * lim)) if lim > 0 else 0.0, 1.0)
        j_speed_limit = over_norm ** 2        # ★修改：显式平方，与 2.0 一致
        self.speed_over_sum += j_speed_limit

        nghbs = o.get("neighbors_state", [])
        if nghbs is not None and len(nghbs) > 0:
            safe_time = 3.0
            obstacle_dist_th = ego_speed * safe_time + 1e-3
            rel_angle_th = np.pi * 40 / 180
            w_dist = 0.05
            di_list = []
            for n in nghbs:
                pos = np.array(n[-1, :2])
                rel = pos - ego_pos
                dist = np.linalg.norm(rel)
                if dist < 1e-3 or dist > obstacle_dist_th:
                    continue
                obs_ang = math.atan2(rel[1], rel[0])
                rel_ang = (obs_ang - ego_heading + np.pi) % (2 * np.pi) - np.pi
                if abs(rel_ang) <= rel_angle_th:
                    di_list.append(dist)
            if di_list:
                di = np.array(di_list)
                j_dist = np.amax(np.exp(-w_dist * di))
                self.dist_to_obs_vals.append(j_dist)
            else:
                self.dist_to_obs_vals.append(0.0)
        else:
            self.dist_to_obs_vals.append(0.0)

        self.steps += 1

    # --------------------------------------------------------------
    def compute(self) -> Costs:

        c = Costs()
        c.dist_to_destination = safe_div(self.dist_goal, self.start_dist, 0.0)
        c.steps = min(self.steps / 200.0, 1.0)
        c.dist_to_obstacles = np.mean(self.dist_to_obs_vals) if self.dist_to_obs_vals else 0.0
        c.jerk_linear = min(self.jerk_mean / 0.9, 1.0)
        c.lane_center_offset = min(safe_div(self.lane_off_sum, self.steps, 0.0) / 1.8, 1.0)
        c.speed_limit = min(safe_div(self.speed_over_sum, self.steps, 0.0), 1.0)
        c.wrong_way = 1.0 if self.wrong else 0.0
        c.collisions = 1.0 if self.collided else 0.0
        c.off_road = 1.0 if self.offroad else 0.0
        return c
