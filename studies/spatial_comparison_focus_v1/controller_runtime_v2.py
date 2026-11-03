"""Only import wiring and offline-report entry differ from frozen controller."""
from pathlib import Path
_S=Path(__file__).resolve().parent
_original=(_S/'controller.py').read_text();assert _original.count('from run import *')==1 and _original.count("('report.py','audit'),('report.py','export')")==1
_patched=_original.replace('from run import *','from runtime_common_v2 import *').replace("('report.py','audit'),('report.py','export')","('report_runtime_v2.py','audit'),('report_runtime_v2.py','export')")
assert _patched.replace('from runtime_common_v2 import *','from run import *').replace("('report_runtime_v2.py','audit'),('report_runtime_v2.py','export')","('report.py','audit'),('report.py','export')")==_original
exec(compile(_patched,str(_S/'controller.py'),'exec'),globals())
