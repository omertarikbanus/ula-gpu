#!/usr/bin/env python3
"""Summarize the fixed-seed comparison, refusing censored recovery results."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

TAGS = ("A_16128x8", "B_16128x16", "C_16128x16_lr3e5")
LABELS = {
    "untrained": "Untrained seed-0 control",
    "A_16128x8": "16128 x 8, LR 1e-5",
    "B_16128x16": "16128 x 16, LR 1e-5",
    "C_16128x16_lr3e5": "16128 x 16, LR 3e-5",
    "icra_epsilon_s0": "Published Epsilon seed 0",
}


def summarize(training, evaluation, checkpoints=None):
    checkpoints = training if checkpoints is None else checkpoints
    outcomes = json.loads((evaluation / "outcomes.json").read_text())
    if outcomes["status"] != "finished":
        raise ValueError("evaluation job did not finish")
    payload = dict(
        seed=0, training_interactions_per_run=84_897_792,
        policy_steps_hz=100, control_hz=500, physics_dt_s=0.001,
        scoring_window_s=5.5, episode_cap_steps=1200, episodes_per_cell=32,
        caveat="One training seed; pooled survival weights the 12 tested cells equally. "
               "This does not establish native transfer, generalization, or a multi-seed ranking.",
        original_censored_scores_excluded=True, rows=[], curves={}, resource_summary={})
    resources = [json.loads(line) for line in (training / "resources.jsonl").read_text().splitlines()]
    library_hashes = set()
    expected_keys = {(d, m) for d in ("fwd", "back", "left", "right", "up", "down")
                     for m in (2.5, 4.0)}
    for tag in ("untrained", *TAGS, "icra_epsilon_s0"):
        score = json.loads((evaluation / f"eval_{tag}.json").read_text())
        cells = score["cells"]
        if {(r["direction"], r["magnitude"]) for r in cells} != expected_keys or len(cells) != 12:
            raise ValueError(f"{tag}: missing/duplicate scoring cells")
        if not score["gates_passed"] or not all(
                all(r["gates"].values()) and r["gates"].get("window_complete")
                and r["completed"] == r["episodes"] == 32 for r in cells):
            raise ValueError(f"{tag}: incomplete or invalid scoring window")
        library_hashes.update(r["library_sha256"] for r in cells)
        row = dict(tag=tag, label=LABELS[tag], cells=cells,
                   survived=sum(r["survived"] for r in cells),
                   episodes=sum(r["episodes"] for r in cells))
        row["survival_pct"] = 100 * row["survived"] / row["episodes"]
        row["by_magnitude"] = {}
        for magnitude in (2.5, 4.0):
            subset = [r for r in cells if r["magnitude"] == magnitude]
            survived = sum(r["survived"] for r in subset)
            episodes = sum(r["episodes"] for r in subset)
            row["by_magnitude"][str(magnitude)] = dict(
                survived=survived, episodes=episodes, survival_pct=100 * survived / episodes)
        if tag in TAGS:
            trained = json.loads((training / f"train_{tag}.json").read_text())
            if trained["interactions"] != payload["training_interactions_per_run"]:
                raise ValueError("training sample budgets differ")
            checkpoint = checkpoints / f"ckpt_{tag}.pt"
            digest = sha256(checkpoint.read_bytes()).hexdigest()
            transported = json.loads((evaluation / "checkpoint_hashes.json").read_text())
            inputs = json.loads((evaluation / "input_manifest.json").read_text())
            if (digest != transported[checkpoint.name]
                    or digest != inputs[f"ula-gpu/bench/eval_inputs/{checkpoint.name}"]):
                raise ValueError("evaluation did not use the final training checkpoint")
            row.update(training_minutes=trained["loop_wall_s"] / 60,
                       training_steps_per_s=trained["env_steps_per_s"],
                       training_kicks=trained["final"]["kicks_total"],
                       checkpoint_sha256=digest)
            all_rows = list(csv.DictReader((training / f"train_{tag}.csv").open()))
            # Equally spaced interaction targets make different rollout lengths
            # comparable on the x-axis, not on their update index.
            selected = []
            for part in range(11):
                target = part * payload["training_interactions_per_run"] / 10
                nearest = min(all_rows, key=lambda r: abs(int(r["interactions"]) - target))
                selected.append({k: float(nearest[k]) for k in
                                 ("interactions", "mean_raw_reward", "fall_rate", "explained_variance")})
            payload["curves"][tag] = selected
            samples = [r for r in resources if r["phase"] == f"train_{tag}"]
            import statistics
            gpu = [[float(x) for x in r["gpu_util_mem_mib_total_mib_watts"].split(",")]
                   for r in samples]
            payload["resource_summary"][tag] = dict(
                sample_count=len(samples), median_gpu_util_pct=statistics.median(x[0] for x in gpu),
                peak_sampled_gpu_mib=max(x[1] for x in gpu),
                median_sampled_power_w=statistics.median(x[3] for x in gpu))
        payload["rows"].append(row)
    if len(library_hashes) != 1:
        raise ValueError("policies were evaluated with different controller builds")
    payload["evaluation_library_sha256"] = library_hashes.pop()
    payload["input_manifest"] = json.loads((evaluation / "input_manifest.json").read_text())
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=Path("bench/artifacts/training"))
    parser.add_argument("--evaluation", type=Path, default=Path("bench/artifacts/evaluation"))
    parser.add_argument("--checkpoints", type=Path, default=Path("bench/artifacts/checkpoints"))
    args = parser.parse_args()
    payload = summarize(args.training, args.evaluation, args.checkpoints)
    output = args.evaluation / "learning_summary.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    for row in payload["rows"]:
        print(f"{row['label']}: {row['survived']}/{row['episodes']} "
              f"({row['survival_pct']:.1f}%)")
    print(output)


if __name__ == "__main__":
    main()
