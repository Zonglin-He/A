"""Serialization-only wrapper preserving locked scientific computation."""
from pathlib import Path
import json, runpy, sys
import numpy as np

original=json.JSONEncoder.default
def scalar_default(self,obj):
    if isinstance(obj,np.generic):return obj.item()
    return original(self,obj)
json.JSONEncoder.default=scalar_default
try:
    script=Path(__file__).with_name('adapt_cached.py');sys.argv=[str(script),*sys.argv[1:]]
    runpy.run_path(str(script),run_name='__main__')
finally:json.JSONEncoder.default=original
