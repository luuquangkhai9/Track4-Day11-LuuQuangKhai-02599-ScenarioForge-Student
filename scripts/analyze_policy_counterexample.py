"""Find a concrete reaction explaining the T=2 TRAIN result; does not select a policy."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from avsim import core, space
from lab11 import estimate, feasible, metrics, search
from scripts.analyze_lab import describe


def main():
    nat, seeds = space.student_variant("2202602599")
    hunt = json.loads((ROOT / "evidence/hunt.json").read_text(encoding="utf-8"))
    q = hunt["bus_bypass"]["proposal"]
    prop = {k: np.array(v) if isinstance(v, list) else v for k, v in q.items()}
    z, b = search.mixture_sample(prop, nat, 400, np.random.default_rng(seeds["search"] + 11))
    params = space.to_params(z, b, nat)
    keep = [i for i, p in enumerate(params) if feasible.is_feasible(p)]
    ps = [params[i] for i in keep]
    base = core.rollout(ps)
    modified = core.rollout(ps, policy={"T": 2.0})
    weights = np.exp(estimate.log_weights(z, b, nat, prop))
    new_hits = [k for k in range(len(ps)) if not metrics.ego_events(base, k) and metrics.ego_events(modified, k)]
    assert new_hits, "No new failure in this TRAIN sample"
    k = max(new_hits, key=lambda j: weights[keep[j]])
    out = dict(source="Extra diagnostic replay of the bus_bypass TRAIN draws from fix; same seed and proposal; not TEST and not an additional probability estimate.",
               training_row=keep[k], policy={"T": 2.0}, params=ps[k], weight=float(weights[keep[k]]),
               baseline=describe(base, k), modified=describe(modified, k))
    f = out["modified"]["events"][0]["f"]
    out["baseline_at_modified_event"] = dict(t=f * core.DT,
        actors={aid: dict(zip(("x", "z", "hx", "hz", "v", "a", "d"),
                              map(float, base.F[k, f, base.ids.index(aid)])))
                for aid in ("ego", "C2", "BUS", "P1")})
    (ROOT / "evidence/policy_counterexample.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
