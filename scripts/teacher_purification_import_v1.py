"""Load the CPU core without vg_tta.__init__'s unrelated eager torch import."""
import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'vg_tta/tastvg_teacher_purification_v1.py'
spec = importlib.util.spec_from_file_location('teacher_purification_cpu_core_v1', path)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
