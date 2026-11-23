#!/usr/bin/env python3
"""Embed the generated replay archive in a Colab notebook, without upload prompts."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import textwrap


BENCH = Path(__file__).resolve().parent
# Keep individual cell execution requests small instead of sending one ~3 MB cell.
CHUNK_CHARS = 192 * 1024


def code(source, *, hide_source=False):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"cellView": "form"} if hide_source else {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def make_notebook(template, archive, validation=False):
    cells = template["cells"]
    upload = [i for i, cell in enumerate(cells) if "files.upload()" in "".join(cell["source"])]
    if len(upload) != 1:
        raise ValueError("expected one file upload cell in the replay notebook template")

    # Never carry over outputs (including old upload widgets) from a saved Colab run.
    for cell in cells:
        if cell["cell_type"] == "code":
            cell.update(execution_count=None, outputs=[])
    cells[0]["source"] = [
        "# ULA Go2 CUDA replay (self-contained)\n",
        "Choose a GPU in **Runtime → Change runtime type** (T4 is enough), "
        "then use **Runtime → Run all**. The code and trace are embedded; "
        "there is no separate upload.\n",
        "\n",
        "Planning, robot dynamics, force problem setup and solve, torque recovery, "
        "and motor output run on CUDA. Task calculations stay on the CPU. This checks replay "
        "correctness, not speed or training."
    ]

    if validation:
        cells[0]["source"] = [
            "# ULA Go2 CUDA controller and Warp validation\n",
            "Choose a T4 GPU under **Runtime → Change runtime type**, then **Run all**. "
            "The code and trace are embedded; no separate upload. This runs the two "
            "NVIDIA solvers in the full controller, checks the recorded trace, then "
            "runs a scripted stand/walk/turn test and a short Warp benchmark. No PPO.\n",
        ]
        cells[1] = code("import subprocess as sp\nfrom pathlib import Path\nimport sys\n")
        cells[-1] = code("""#@title Check the full controller and the Warp closed loop
log_path = repo / 'go2_validation.log'
with sp.Popen([sys.executable, '-u', str(repo / 'bench/run_go2_colab.py'), '--repo', str(repo)],
              stdout=sp.PIPE, stderr=sp.STDOUT, text=True) as p, log_path.open('w') as log:
    for line in p.stdout:
        log.write(line)
        print(line, end='', flush=True)
print(f'Full log: {log_path}')
if p.returncode:
    raise RuntimeError(f'Go2 validation returned {p.returncode}')
""")

    encoded = base64.b64encode(archive).decode("ascii")
    chunks = [encoded[i:i + CHUNK_CHARS] for i in range(0, len(encoded), CHUNK_CHARS)]
    payload = []
    for index, chunk in enumerate(chunks):
        title = f"#@title Embedded replay bundle ({index + 1}/{len(chunks)})\n"
        init = "_bundle_chunks = {}\n" if index == 0 else ""
        lines = "".join(f"    {part!r}\n" for part in textwrap.wrap(chunk, width=1024))
        payload.append(code(f"{title}{init}_bundle_chunks[{index}] = (\n{lines})\n", hide_source=True))

    digest = hashlib.sha256(archive).hexdigest()
    extract = code(f"""#@title Extract the embedded replay bundle
import base64
import hashlib
import io
from tempfile import mkdtemp
import tarfile

archive = base64.b64decode(''.join(_bundle_chunks[i] for i in range({len(chunks)})), validate=True)
if hashlib.sha256(archive).hexdigest() != '{digest}':
    raise ValueError('The embedded bundle failed its checksum.')
runtime_dir = Path('/content')
run_dir = Path(mkdtemp(prefix='ula-replay-', dir=runtime_dir if runtime_dir.is_dir() else None))
with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as bundle:
    bundle.extractall(run_dir, filter='data')
repo = run_dir / 'ula-gpu'
if not (repo / 'data/go2_reference.jsonl').is_file():
    raise FileNotFoundError('Missing bundled reference trace.')
print('Replay bundle extracted. The next cell builds and runs the test.')
""")
    i = upload[0]
    cells[i:i + 1] = [*payload, extract]
    return template


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validation", action="store_true", help="run the vendor solvers and Warp validation")
    args = parser.parse_args()
    args.archive = args.archive or BENCH / "artifacts/bundles/ula-go2-vendor.tar.gz"
    args.output = args.output or BENCH / "artifacts/notebooks" / (
        "colab_go2_validation.ipynb" if args.validation else "colab_cuda_replay_standalone.ipynb")
    template_path = BENCH / "colab_cuda_replay.ipynb"
    if args.output.resolve() in (template_path.resolve(), args.archive.resolve()):
        parser.error("the standalone notebook must not replace the template or archive")
    notebook = make_notebook(json.loads(template_path.read_text(encoding="utf-8")), args.archive.read_bytes(),
                             args.validation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
