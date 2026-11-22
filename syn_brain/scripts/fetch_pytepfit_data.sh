#!/usr/bin/env bash
# Fetch PyTepFit's real TMS-EEG file (only_high_trial.mat, 11.5 MB) into syn_brain/external/pytepfit/.
#
# The anatomy files (leadfield, connectome, distances) are already in git there. Only the real-data
# file is fetched, because it is large; it is needed only by scripts/calibrate_teacher.py.
#
# Usage (from anywhere):  bash syn_brain/scripts/fetch_pytepfit_data.sh
# Copies from a local PyTepFit clone at ~/projects/PyTepFit if one exists, else downloads from GitHub.

set -euo pipefail
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/../external/pytepfit" && pwd)"
F=only_high_trial.mat
URL="https://raw.githubusercontent.com/GriffithsLab/PyTepFit/66b94e488c82478e6f302015b6cd8e8d9d33792a/data/$F"
SIZE=11513885

if [[ -f "$DEST/$F" && $(wc -c < "$DEST/$F") -eq $SIZE ]]; then
  echo "$F already present in $DEST"
  exit 0
fi
if [[ -f "$HOME/projects/PyTepFit/data/$F" ]]; then
  echo "copying $F from local PyTepFit clone"
  cp "$HOME/projects/PyTepFit/data/$F" "$DEST/$F"
else
  echo "downloading $F (11.5 MB) from GitHub"
  curl -fL --retry 3 -o "$DEST/$F.part" "$URL"
  mv "$DEST/$F.part" "$DEST/$F"
fi
got=$(wc -c < "$DEST/$F")
if [[ $got -ne $SIZE ]]; then
  echo "ERROR: $F has $got bytes, expected $SIZE" >&2
  exit 1
fi
echo "ok: $DEST/$F"
