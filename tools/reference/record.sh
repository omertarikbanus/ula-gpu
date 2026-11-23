#!/usr/bin/env bash
# Read Git objects only; build and run on a disposable source snapshot.
set -euo pipefail
if (( $# < 4 || $# > 6 )); then
  echo "Usage: $0 ULA_REPO BAZEL_9_BINARY BAZEL_REPOSITORY_CACHE OUTPUT_DIR [EP0_TICKS=300] [EP1_TICKS=100]" >&2
  exit 2
fi
repo=$(realpath -- "$1")
bazel=$(realpath -- "$2")
cache=$(realpath -- "$3")
out=$(realpath -m -- "$4")
case "$out/" in
  "$repo/"*) echo "Choose an output directory outside the live ULA checkout" >&2; exit 2 ;;
esac
here=$(cd -- "$(dirname -- "$0")" && pwd)
ref=29c3dfccde133ffe3bbf56608bd4b0a96777de2e
scratch=$(mktemp -d /tmp/opencode/ula-replay.XXXXXX)
mkdir -p "$scratch/src" "$scratch/bazel/home" "$out"
git -C "$repo" archive "$ref" | tar -x -C "$scratch/src"
for path in third-party/rtrobot third-party/envpool; do
  revision=$(git -C "$repo" rev-parse "$ref:$path")
  git -C "$repo/$path" archive "$revision" | tar -x -C "$scratch/src/$path"
done
git -C "$scratch/src" apply --whitespace=nowarn "$here/recorder.patch"
echo "Disposable reference source and build cache: $scratch"

docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp/bazel_work/home \
  -v "$scratch/src:/app" -v "$scratch/bazel:/tmp/bazel_work" \
  -v "$out:/output" -v "$bazel:/usr/local/bin/bazel9:ro" \
  -v "$cache:/bazel_repo_cache:ro" -w /app ula:dev bash -ec '
    bazel9 --output_user_root=/tmp/bazel_work/out build \
      --repository_cache=/bazel_repo_cache --repo_contents_cache=/tmp/bazel_work/rcc \
      --jobs=3 //src/controller:reference_recorder
    for run in a b; do
      bazel-bin/src/controller/reference_recorder --output "/output/reference-$run.jsonl" \
        --ep0-ticks "$1" --ep1-ticks "$2"
    done
  ' _ "${5:-300}" "${6:-100}"
cmp -- "$out/reference-a.jsonl" "$out/reference-b.jsonl"
