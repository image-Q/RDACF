# -*- coding: utf-8 -*-

import os
import csv
import json
import numpy as np
from metric.costs import CostFuncsV1
from metric.formula import costs_to_score

LOG_DIR = "test_log/Test"
OUT_DIR = os.path.join(LOG_DIR, "metrics")
os.makedirs(OUT_DIR, exist_ok=True)


def read_csv_episode(path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        data = [row for row in reader]
    return data


def _to_float(v):
    if isinstance(v, (float, int)):
        return float(v)
    if isinstance(v, str):
        v = v.strip()
        if v.startswith("Heading(") and v.endswith(")"):
            v = v[len("Heading("):-1]
        try:
            return float(v)
        except:
            return 0.0
    return 0.0


def evaluate_episode(csv_path):
    data = read_csv_episode(csv_path)
    costfunc = CostFuncsV1()
    goal_pos = np.array([_to_float(data[0]["goal_x"]), _to_float(data[0]["goal_y"])])

    for row in data:
        ego_state = np.array([
            [_to_float(row["x"]), _to_float(row["y"]),
             _to_float(row["heading"]), _to_float(row["speed"])]
        ])

        neighbors_state = []
        for i in range(5):
            nx = float(row.get(f"neighbor{i}_x", 0.0))
            ny = float(row.get(f"neighbor{i}_y", 0.0))
            neighbors_state.append(np.array([[nx, ny, 0.0, 0.0, 0.0]]))
        neighbors_state = np.array(neighbors_state)


        class Events:
            pass
        ev = Events()
        ev.wrong_way = bool(int(row["wrong_way"]))
        ev.collisions = [1] if int(row["collision"]) else []
        ev.off_road = bool(int(row["off_road"]))

        # observation dict
        o = {
            "ego_state": ego_state,
            "goal_pos": goal_pos,
            "lane_center_offset": float(row["lane_center_offset"]),
            "speed_limit": float(row["speed_limit"]),
            "neighbors_state": neighbors_state,
            "events": ev,
        }

        costfunc.step(o, dt=0.1)


    costs = costfunc.compute()
    score = costs_to_score(costs)
    return costs, score

def main():
    csv_files = [f for f in os.listdir(LOG_DIR) if f.endswith(".csv") and not f.startswith("test_log")]
    results = []

    print(f"🔍 Found {len(csv_files)} episode logs in {LOG_DIR}")

    for csv_file in csv_files:
        path = os.path.join(LOG_DIR, csv_file)
        print(f"➡ Evaluating {csv_file} ...")
        costs, score = evaluate_episode(path)

        ep_name = os.path.splitext(csv_file)[0]
        out_json = os.path.join(OUT_DIR, f"{ep_name}_metrics.json")

        with open(out_json, "w") as f:
            json.dump({"costs": costs.__dict__, "score": score}, f, indent=2)

        results.append({
            "episode": ep_name,
            "dist_to_destination": round(score["dist_to_destination"], 4),
            "time": round(score["time"], 4),
            "humanness_error": round(score["humanness_error"], 4),
            "rule_violation": round(score["rule_violation"], 4),
            "overall": round(score["overall"], 4)
        })


    out_csv = os.path.join(OUT_DIR, "metrics_summary.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    mean_score = {k: np.mean([r[k] for r in results if k != "episode"]) for k in results[0] if k != "episode"}
    print("\n=== 平均指标 ===")
    for k, v in mean_score.items():
        print(f"{k:20s}: {v:.4f}")

    print(f"\n✅ 已生成每个 episode 的 JSON 与汇总文件：{OUT_DIR}/metrics_summary.csv")


if __name__ == "__main__":
    main()
