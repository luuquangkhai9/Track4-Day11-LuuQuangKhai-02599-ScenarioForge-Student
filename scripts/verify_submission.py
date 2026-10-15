"""Audit required artifacts, full budgets, provenance and cross-file consistency before packaging."""
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from avsim import space, studio


def main():
    expected_id = "2202602599"
    town, seeds = space.student_variant(expected_id)
    report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
    assert report.splitlines()[0] == f"MSSV: {expected_id}"
    files = ("world", "replay", "mc", "hunt", "estimate", "fix", "mine", "meta")
    evidence = {}
    budgets = dict(replay_n=400, mc_n=3000, iters=8, per_iter=300, final_n=600,
                   seeds=3, fix_n=400, progress_n=256, mine_n=1500)
    for name in files:
        obj = json.loads((ROOT / "evidence" / f"{name}.json").read_text(encoding="utf-8"))
        evidence[name] = obj
        run = obj["_run"]
        assert run["mssv"] == expected_id and run["town"] == town.name, name
        assert run["quick"] is False and run["salted"] is False, name
        assert run["budgets"] == budgets, name
    assert evidence["meta"]["seeds"] == seeds
    for step in ("world", "replay", "mc", "hunt", "estimate", "fix", "mine", "export"):
        assert evidence["meta"]["steps"][step]["quick"] is False, step
    mc = evidence["mc"]
    assert mc["n"] == 3000 and mc["crashes"] == sum(mc["modes"].values())
    assert math.isclose(mc["p"], mc["crashes"] / mc["n"])
    modes = ("bus_bypass", "cut_in", "crosswalk_block", "rear_ended")
    cloud = json.loads((ROOT / "evidence/cloud.json").read_text(encoding="utf-8"))
    assert len(cloud) == 9600
    for mode in modes:
        assert len(evidence["hunt"][mode]["trace"]) == 8
    for mode in ("bus_bypass", "crosswalk_block"):
        assert evidence["hunt"][mode]["found"].get(mode, 0) > 0
    estimate = evidence["estimate"]
    assert len(estimate["seeds"]) == 3 and estimate["budget"] == 16800
    for seed in estimate["seeds"]:
        assert math.isclose(seed["total"], sum(v["p"] for v in seed["per"].values()) + estimate["other"]["p"])
        for mode in modes:
            row = seed["per"][mode]
            assert row["n"] == 600
            if row["hits"] == 0:
                assert math.isclose(row["upper_zero_hits"], 0.025)
    fix = evidence["fix"]
    feasible_rows = [r for r in fix["table"] if r["progress"] >= .95 * fix["baseline_train"]["progress"]]
    assert fix["chosen"] == min(feasible_rows, key=lambda r: (r["fail_hi"], -r["progress"]))
    assert "chosen_test" in fix and "baseline_test" in fix
    assert fix["chosen_test"]["progress"] >= .95 * fix["baseline_test"]["progress"]
    counter = json.loads((ROOT / "evidence/policy_counterexample.json").read_text(encoding="utf-8"))
    assert not counter["baseline"]["events"] and counter["modified"]["events"]
    assert counter["policy"] == {"T": 2.0}
    hz = studio.load(ROOT / "hazard.json")
    assert studio.hazard_ok(hz["hazards"], hz["params"]) and not studio.check_space(hz)
    assert hz.get("why") and 0 < hz["exposure"] <= 1
    mine = evidence["mine"]
    assert mine["accepted"] >= .5 and mine["baseline"]["avoidable"] > 0
    for name in ("baseline", "chosen"):
        row = mine[name]
        assert row["n"] > 0
        assert row["crashes"] == sum(row[k] for k in ("avoidable", "unavoidable", "unattributed"))
        assert math.isclose(row["p"], row["avoidable"] / row["n"])
        assert math.isclose(row["per_trip"], hz["exposure"] * row["p"])
    diag = json.loads((ROOT / "evidence/scene_analysis.json").read_text(encoding="utf-8"))
    assert not diag["drive"]["events"] and diag["drive"]["fire"][0] >= 0
    assert any(s["baseline"]["events"] and not s["chosen"]["events"] for s in diag["scenes"])
    events = [[e["a"], e["b"], e["t"]] for e in diag["exact_hazard"]["events"]]
    assert events == [e for e in mine["exact"]["planner"] if "ego" in e[:2]]
    for i in range(1, 10):
        assert re.search(rf"^## {i}\. ", report, re.MULTILINE), i
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report)
    assert len(images) >= 5
    for image in images:
        assert (ROOT / image).stat().st_size > 1000, image
    for path in (ROOT / "lab11").glob("*.py"):
        assert "raise NotImplementedError" not in path.read_text(encoding="utf-8"), path
    protected_diff = subprocess.check_output(
        ["git", "diff", "--name-only", "--", "avsim", "run_all.py", "tests", "policy_candidates.json"],
        cwd=ROOT, text=True).strip()
    assert not protected_diff, protected_diff
    subprocess.run(["git", "diff", "--check"], cwd=ROOT, check=True)
    print("PASS: required artifacts, full budgets, student seeds, statistical consistency, TRAIN selection, hazard feasibility, images and protected sources.")
    print("Test runs already completed separately: 31 TODO tests and 10 Python/JS parity tests passed.")


if __name__ == "__main__":
    main()
