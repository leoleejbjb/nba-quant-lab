#!/usr/bin/env bash
# Package the repository for hand-off: a git bundle (full history) and a source tarball (HEAD, no data).
#   scripts/package.sh <out_dir>
set -euo pipefail
out="${1:-dist}"
mkdir -p "$out"
git bundle create "$out/nba-quant-lab.bundle" --all
git archive --format=tar.gz --prefix=nba-quant-lab/ -o "$out/nba-quant-lab-src.tar.gz" HEAD
git bundle verify "$out/nba-quant-lab.bundle" >/dev/null && echo "bundle ok"
ls -la "$out"
