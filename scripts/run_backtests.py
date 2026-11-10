"""Walk-forward simulation backtests.

    python scripts/run_backtests.py --seasons 2023-24 --modes none,team [--n-sims 1500] [--limit N]

Writes data/backtest/sim_<mode>_<season>_{players,teams}.parquet. Every input is pregame
(see src/quantlab/sim/inputs.py); the simulator parameters were fixed on 2023-24.
"""
import argparse
import time

from quantlab.eval.backtest import run_backtest
from quantlab.sim.inputs import InputStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2023-24")
    ap.add_argument("--modes", default="none")
    ap.add_argument("--n-sims", type=int, default=1500)
    ap.add_argument("--blend-weight", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    store = InputStore()
    for season in a.seasons.split(","):
        ids = store.games[store.games["season"] == season].sort_values("game_date").index.tolist()
        if a.limit:
            ids = ids[: a.limit]
        for mode in a.modes.split(","):
            env = "blend" if mode.startswith("blend") else mode
            tag = f"sim_{mode}_{season}"
            t0 = time.time()
            run_backtest(ids, n_sims=a.n_sims, environment=env, env_weight=a.blend_weight, store=store, tag=tag)
            print(f"[{tag}] {len(ids)} games in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
