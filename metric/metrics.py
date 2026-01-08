# -*- coding: utf-8 -*-


import json, pathlib, time, csv
from .costs import CostFuncsV1
from .formula import costs_to_score

class MetricsV1:

    def __init__(self, dt=0.1):
        self.dt = dt
        self.costs = CostFuncsV1()

    def reset(self):
        self.costs.reset()

    def step(self, obs, done):
        if not obs:
            return
        aid = list(obs.keys())[0]
        self.costs.step(obs[aid], dt=self.dt)

    def score(self):
        c = self.costs.compute()
        return costs_to_score(c)

    def save(self, out_dir="test_log/metrics", scenario="unknown", episode="0"):
        out_path = pathlib.Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        scores = self.score()
        ts = time.strftime("%Y%m%d_%H%M%S")
        json_path = out_path / f"{ts}_{scenario}_ep{episode}.json"
        csv_path = out_path / f"{ts}_{scenario}_ep{episode}.csv"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Metric", "Score"])
            for k, v in scores.items():
                w.writerow([k, v])

        print(f"[MetricsV1] Saved results to {json_path}")
