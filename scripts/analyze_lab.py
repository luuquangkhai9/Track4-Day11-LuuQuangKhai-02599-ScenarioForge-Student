"""Derive scene explanations from the lab's saved evidence; never changes the pipeline or its seeds."""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from avsim import core, studio, viewer as viewer_export
from lab11.metrics import ego_events


def read(name):
    return json.loads((ROOT / "evidence" / f"{name}.json").read_text(encoding="utf-8"))


def describe(R, k):
    ev = ego_events(R, k)
    f = ev[0]["f"] if ev else core.NF - 1
    frames = sorted(set(max(0, f - d) for d in (80, 40, 20, 0)))
    shots = []
    for frame in frames:
        actors = {}
        for aid in ["ego", "C2", "TG", "BUS", "P1"] + [i for i in R.ids if i.startswith("H")]:
            q = R.F[k, frame, R.ids.index(aid)]
            actors[aid] = dict(zip(("x", "z", "hx", "hz", "v", "a", "d"), map(float, q)))
        leader = int(R.EGO[k, frame, 3])
        shots.append(dict(t=frame * core.DT, actors=actors,
                          sees_P1=bool(R.EGO[k, frame, 2]),
                          leader=R.ids[leader] if leader >= 0 else None,
                          ego_mode=int(R.EGO[k, frame, 0])))
    e = R.F[k, :f + 1, core.EGO]
    return dict(events=ev, snapshots=shots,
                progress_to_event=float(np.hypot(np.diff(e[:, 0]), np.diff(e[:, 1])).sum()))


def main():
    viewer, fix = read("viewer"), read("fix")
    chosen = fix["chosen"]["policy"]
    runs = [r for r in viewer["runs"] if r["label"] in ("bus_bypass", "cut_in", "crosswalk_block", "rear_ended")]
    params = [r["params"] for r in runs]
    base = core.rollout(params)
    new = core.rollout(params, policy=chosen)
    scenes = [dict(id=r["id"], mode=r["label"], params=r["params"],
                   baseline=describe(base, k), chosen=describe(new, k)) for k, r in enumerate(runs)]
    avoided = next((s for s in scenes if s["baseline"]["events"] and not s["chosen"]["events"]), None)
    if avoided:
        viewer_export.export(ROOT / "evidence" / "fix_comparison.json",
                             [dict(id="FIX-01", label=avoided["mode"], params=avoided["params"], outcome="safe")],
                             policy=chosen, baseline_policy={}, title="Policy fix vs baseline - MSSV 2202602599")
    replay = read("replay")["replay_only"][0]
    p = replay["params"]
    human_log = core.rollout(p, ego="human")
    rep, rea = core.rollout(p, others="replay", log=human_log), core.rollout(p)
    doc = studio.load(ROOT / "hazard.json")
    hazard = core.rollout(doc["params"], policy=doc["policy"], hazards=doc["hazards"])
    avoid, f0, foreknown = studio.oracle_avoids([doc["params"]], hazard, doc["hazards"])
    out = dict(
        source="Derived from viewer.json, replay.json, fix.json, hazard.json and drive_scene.json; extra diagnostic rollouts, not new Monte Carlo evidence.",
        scenes=scenes,
        replay_example=dict(scenario=replay["scenario"], replay=describe(rep, 0), reactive=describe(rea, 0)),
        exact_hazard=dict(**describe(hazard, 0), fire=hazard.fire[0].tolist(),
                          oracle_brake_frame=int(f0[0]), avoidable=bool(avoid[0]),
                          avoidable_foreknown=bool(foreknown[0])),
    )
    driven_doc = studio.load(ROOT / "evidence/drive_scene.json")
    assert driven_doc["params"] == doc["params"] and driven_doc["hazards"] == doc["hazards"]
    pedal = np.asarray(driven_doc["egoAcc"], dtype=float)
    driven = core.rollout(driven_doc["params"], policy=driven_doc["policy"], ego="script",
                          ego_acc=pedal, hazards=driven_doc["hazards"])
    braking = np.flatnonzero(pedal < 0)
    out["drive"] = dict(**describe(driven, 0), fire=driven.fire[0].tolist(),
                        brake_start=float(braking[0] * core.DT),
                        brake_end=float((braking[-1] + 1) * core.DT),
                        brake_acceleration=float(pedal[braking[0]]),
                        source="Keyboard control recorded through chapter 10 after replaying the scene; not a blinded human reaction trial.")
    path = ROOT / "evidence" / "scene_analysis.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)
    for s in scenes:
        print(s["id"], s["mode"], s["baseline"]["events"], "chosen:", s["chosen"]["events"])
    print("Exact hazard:", out["exact_hazard"]["events"], "avoidable:", bool(avoid[0]))


if __name__ == "__main__":
    main()
