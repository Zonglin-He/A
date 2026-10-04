"""Load the geometry module without vg_tta's package-level model imports."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('spatial_guided_temporal_core',Path(__file__).resolve().parents[1]/'vg_tta/tastvg_spatial_guided_temporal_p0_v1.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
