"""Run pinned audit components against this study's own paths and arms."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = HERE.parent / "3eed_hispatial_observation_audit_20261121"
QWEN = HERE.parent / "3eed_frozen_qwen_metric_grounding_20261120"
PRIVATE = HERE / "private"
OUT = HERE / "results"
PROTOCOL = HERE / "PROTOCOL.json"
ARMS = ("moge", "global_scale", "prior_depth", "wrong_geometry", "oracle_depth")

sys.path.insert(0, str(BASE))
import common

# Bind the previous study's audited functions before importing their modules.
# This changes only process-local configuration, never its saved files.
common.PRIVATE = PRIVATE
common.OUT = OUT
common.PROTOCOL = PROTOCOL
common.ARMS = ARMS
from common import ASSETS, PLATFORMS, SIDE, digest, key, read, write
