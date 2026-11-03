"""Unique-module loader: preserve frozen numerical code and avoid run shadowing."""
from pathlib import Path
import importlib.util,json,hashlib,sys
_S=Path(__file__).resolve().parent;_R=_S.parents[1]
_p=json.loads((_R/'outputs/spatial_comparison_focus_v1/execution_v2_plan.json').read_text())
for _f,_h in _p['code_sha256'].items():assert hashlib.sha256((_R/_f).read_bytes()).hexdigest()==_h,_f
_spec=importlib.util.spec_from_file_location('_spatial_comparison_focus_frozen_numeric_v1',_S/'run.py');_mod=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_mod;_spec.loader.exec_module(_mod);_mod.check()
globals().update({_k:_v for _k,_v in vars(_mod).items()if not _k.startswith('_')})
