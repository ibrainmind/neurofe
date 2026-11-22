#!/usr/bin/env bash
# Re-run the single-node virtual-TMS study end to end and rebuild the PDF report.
#
# Usage (from anywhere):
#   bash syn_brain/scripts/run_single_node_tms.sh            # everything (~45 min on 4 cores)
#   NPARTS=8 bash syn_brain/scripts/run_single_node_tms.sh   # more parallel workers on a bigger server
#   STEP=report bash syn_brain/scripts/run_single_node_tms.sh  # only rebuild figures + PDF from saved results
#
# Environment variables:
#   PYTHON   python interpreter with neurofe's dependencies (default: <repo>/.venv/bin/python, else python3)
#   NPARTS   parallel training processes (default 4; one CPU core each)
#   ITERS    Adam iterations per model (default 300, as in the report)
#   STEP     all | analyse | train | report   (default all)
#   SYN_BRAIN_PYTEPFIT_DATA  folder with PyTepFit's data (default ~/projects/PyTepFit/data)
#
# Outputs: syn_brain/results/single_node_tms/ (data, figures), syn_brain/logs/ (logs),
#          syn_brain/reports/single_node_tms_study.pdf

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
PYTHON="${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}"
NPARTS="${NPARTS:-4}"
ITERS="${ITERS:-300}"
STEP="${STEP:-all}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
LOGS=syn_brain/logs
mkdir -p "$LOGS"

# Keep a laptop awake during long runs (macOS); no-op elsewhere.
KEEPAWAKE=""
command -v caffeinate >/dev/null 2>&1 && KEEPAWAKE="caffeinate -i"

run_parts() {   # $1 = experiments for --only ("" = E1-E7), $2 = log tag
  local pids=()
  for ((p = 0; p < NPARTS; p++)); do
    $KEEPAWAKE "$PYTHON" -m syn_brain.scripts.single_node_tms train --iters "$ITERS" \
      ${1:+--only "$1"} --part "$p" --nparts "$NPARTS" > "$LOGS/sn_tms_${2}_part$p.out" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
}

echo "python: $PYTHON  parts: $NPARTS  iters: $ITERS  step: $STEP"
if [[ "$STEP" == all || "$STEP" == analyse ]]; then
  echo "[1/4] Fisher / feature / tonic-drive analysis (~2 min)"
  "$PYTHON" -m syn_brain.scripts.single_node_tms analyse > "$LOGS/sn_tms_analyse.out" 2>&1
fi
if [[ "$STEP" == all || "$STEP" == train ]]; then
  rm -f syn_brain/results/single_node_tms/train*_part*of*.npz   # old parts would be merged with new ones
  echo "[2/4] training E1-E7, 372 models in $NPARTS parts (~25 min on 4 cores)"
  run_parts "E1,E2,E3,E4,E5,E6,E7" main
  echo "[3/4] training E8 temporal hold-out, 30 models (~20 min)"
  run_parts "E8" e8
fi
if [[ "$STEP" == all || "$STEP" == report ]]; then
  echo "[4/4] figures, summary and PDF"
  "$PYTHON" -m syn_brain.scripts.single_node_tms_figures
  "$PYTHON" -m syn_brain.scripts.single_node_tms_report
fi
echo "done"
