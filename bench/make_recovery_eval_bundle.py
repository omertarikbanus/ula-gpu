#!/usr/bin/env python3
"""Add verified final training checkpoints and published seed-0 inputs to a bundle."""

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ula_gpu.icra_inference import epsilon_paths, verify_epsilon_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    published = epsilon_paths(args.policy_root, 0)
    verify_epsilon_files(0, *published)
    files = {
        f"ula-gpu/bench/eval_inputs/{name}": args.checkpoints / name for name in
        ("ckpt_A_16128x8.pt", "ckpt_B_16128x16.pt", "ckpt_C_16128x16_lr3e5.pt")
    }
    files.update({f"ula-gpu/bench/eval_inputs/epsilon_s0/{p.name}": p for p in published})
    expected = json.loads((args.checkpoints / "checkpoint_hashes.json").read_text())
    hashes = {}
    for name, path in files.items():
        hashes[name] = sha256(path.read_bytes()).hexdigest()
        if path.suffix == ".pt" and hashes[name] != expected[path.name]:
            raise ValueError(f"not the verified final checkpoint: {path}")
    temporary = args.output.with_suffix(".tmp")
    with tarfile.open(args.base) as source, tarfile.open(temporary, "w:gz") as target:
        for member in source:
            target.addfile(member, source.extractfile(member) if member.isfile() else None)
        for name, path in files.items():
            target.add(path, arcname=name)
        payload = json.dumps(hashes, indent=2).encode()
        info = tarfile.TarInfo("ula-gpu/bench/eval_inputs/manifest.json")
        info.size = len(payload)
        target.addfile(info, io.BytesIO(payload))
    with tarfile.open(temporary) as archive:
        for name, expected_hash in hashes.items():
            if sha256(archive.extractfile(name).read()).hexdigest() != expected_hash:
                raise ValueError(f"bundle input mismatch: {name}")
    temporary.replace(args.output)
    print(f"Verified {len(files)} checkpoint inputs in {args.output}", flush=True)


if __name__ == "__main__":
    main()
