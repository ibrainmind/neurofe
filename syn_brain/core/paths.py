"""Folder layout for the study: code lives in core/ and scripts/, outputs in data/, results/, logs/."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

SYN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYN_ROOT.parent
DATA_DIR = SYN_ROOT / "data"
RESULTS_DIR = SYN_ROOT / "results"
LOGS_DIR = SYN_ROOT / "logs"

# PyTepFit's bundled anatomy and TMS-EEG data. Override with SYN_BRAIN_PYTEPFIT_DATA.
PYTEPFIT_DATA = Path(os.environ.get(
    "SYN_BRAIN_PYTEPFIT_DATA",
    str(Path.home() / "projects" / "PyTepFit" / "data")))

# Make `import neurofe` work when scripts are run from anywhere.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def get_logger(name: str) -> logging.Logger:
    """Logger that writes to the console and to logs/<name>_<timestamp>.log."""
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(LOGS_DIR / f"{name}_{stamp}.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_to_builtin)


def _to_builtin(x):
    import numpy as np
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    raise TypeError(f"Not JSON serializable: {type(x)}")
