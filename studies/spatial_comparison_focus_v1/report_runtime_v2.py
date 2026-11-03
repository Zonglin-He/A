"""Same frozen audit/export functions with uniquely loaded numerical module."""
from pathlib import Path
_S=Path(__file__).resolve().parent;_original=(_S/'report.py').read_text();assert _original.count('from run import *')==1
_patched=_original.replace('from run import *','from runtime_common_v2 import *');assert _patched.replace('from runtime_common_v2 import *','from run import *')==_original
exec(compile(_patched,str(_S/'report.py'),'exec'),globals())
