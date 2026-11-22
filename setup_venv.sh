#!/usr/bin/env bash
# Create (or update) the repo's .venv with everything neurofe and syn_brain need.
#
# Usage, from anywhere:
#   bash setup_venv.sh            # create/update .venv, then run a quick check
#   source setup_venv.sh          # same, and leave the environment activated in this shell
#
# Options (environment variables):
#   PYTHON=python3.11             interpreter used to create the venv (default: newest python3.x found, >= 3.9)
#   TORCH_INDEX_URL=...           custom PyTorch wheel index, e.g. CPU-only on a server without GPU:
#                                   TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu bash setup_venv.sh
#   RECREATE=1                    delete and rebuild .venv from scratch
#   SKIP_TESTS=1                  skip the quick pytest check at the end
#
# PyTepFit's data (anatomy, leadfield, real TMS-EEG) is not in this repo. syn_brain looks for it in
# ~/projects/PyTepFit/data, or set SYN_BRAIN_PYTEPFIT_DATA=/path/to/PyTepFit/data.

_sourced=0
[[ "${BASH_SOURCE[0]}" != "$0" ]] && _sourced=1

_setup_venv() {
  local repo py v
  repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "$repo" || return 1

  # 1) pick an interpreter (neurofe declares >= 3.10; 3.9 also works for the code in this repo)
  py="${PYTHON:-}"
  if [[ -z "$py" ]]; then
    for cand in python3.12 python3.11 python3.10 python3 python3.9; do
      if command -v "$cand" >/dev/null 2>&1 &&
         "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
        py="$cand"; break
      fi
    done
  fi
  if [[ -z "$py" ]]; then
    echo "ERROR: no Python >= 3.9 found. Install one or set PYTHON=/path/to/python3.x" >&2
    return 1
  fi
  v="$("$py" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  echo "[1/5] using $py (Python $v)"

  # 2) create the venv
  if [[ "${RECREATE:-0}" == 1 && -d .venv ]]; then
    echo "      removing existing .venv (RECREATE=1)"
    rm -rf .venv
  fi
  if [[ ! -x .venv/bin/python ]]; then
    echo "[2/5] creating .venv"
    "$py" -m venv .venv || { echo "ERROR: venv creation failed (on Debian/Ubuntu: apt install python$v-venv)" >&2; return 1; }
  else
    echo "[2/5] .venv exists, updating it"
  fi
  .venv/bin/python -m pip install --quiet --upgrade pip setuptools wheel || return 1

  # 3) PyTorch (separate step so a custom wheel index can be used)
  echo "[3/5] installing PyTorch"
  if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
    .venv/bin/pip install --quiet "torch>=2.0" --index-url "$TORCH_INDEX_URL" || return 1
  else
    .venv/bin/pip install --quiet "torch>=2.0" || return 1
  fi

  # 4) everything else: neurofe's dependencies + syn_brain (simulation, figures, PDF reports, tests)
  echo "[4/5] installing scientific stack"
  local rl="reportlab>=4.0"
  # reportlab >= 4.1 refuses to import on Python < 3.10; 4.0.x works there
  .venv/bin/python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || rl="reportlab>=4.0,<4.1"
  .venv/bin/pip install --quiet "numpy>=1.21,<2.0" "scipy>=1.7" "scikit-learn>=1.0" "pandas>=1.3" \
    "matplotlib>=3.4" "$rl" "pillow>=8.0" "pytest>=7.0" || return 1
  # neurofe itself in editable mode (needs Python >= 3.10 per pyproject); otherwise PYTHONPATH is enough
  if .venv/bin/python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    .venv/bin/pip install --quiet --no-deps -e . || echo "      (editable install skipped; PYTHONPATH=. works too)"
  else
    echo "      Python < 3.10: skipping 'pip install -e .'; scripts use PYTHONPATH=. instead"
  fi

  # 5) quick check
  echo "[5/5] checking"
  PYTHONPATH="$repo" .venv/bin/python - <<'EOF' || return 1
import numpy, scipy, torch, matplotlib, reportlab
import neurofe
from syn_brain.core import node
print(f"      torch {torch.__version__} (CUDA available: {torch.cuda.is_available()}), numpy {numpy.__version__}, "
      f"threads {torch.get_num_threads()}")
from syn_brain.core.paths import PYTEPFIT_DATA
print(f"      PyTepFit data: {PYTEPFIT_DATA} -> {'found' if PYTEPFIT_DATA.exists() else 'MISSING (set SYN_BRAIN_PYTEPFIT_DATA)'}")
EOF
  if [[ "${SKIP_TESTS:-0}" != 1 ]]; then
    if [[ -d "${SYN_BRAIN_PYTEPFIT_DATA:-$HOME/projects/PyTepFit/data}" ]]; then
      echo "      running syn_brain tests (~1 min)"
      PYTHONPATH="$repo" .venv/bin/python -m pytest syn_brain/tests -q || echo "      WARNING: tests failed (see above)"
    else
      echo "      skipping tests: PyTepFit data not found"
    fi
  fi

  echo
  echo "Done. Next:"
  echo "  source .venv/bin/activate && export PYTHONPATH=$repo"
  echo "  bash syn_brain/scripts/run_single_node_tms.sh        # full single-node study (~45 min on 4 cores)"
}

if [[ $_sourced == 1 ]]; then
  # sourced: never 'exit' (that would close the user's shell); activate on success
  if _setup_venv; then
    # shellcheck disable=SC1091
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.venv/bin/activate"
    export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)${PYTHONPATH:+:$PYTHONPATH}"
    echo "(.venv activated, PYTHONPATH set)"
  fi
  unset -f _setup_venv; unset _sourced
else
  set -o pipefail
  _setup_venv || exit 1
fi
