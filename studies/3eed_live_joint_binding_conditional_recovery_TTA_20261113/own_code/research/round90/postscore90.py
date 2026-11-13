"""Sequential sealed scalar join, independent audit and descriptive reports."""
from pathlib import Path
import subprocess,sys
H=Path(__file__).resolve().parent
for name in ['score90.py','audit_scalar90.py','analyze90.py','cohort90.py','report90.py','findings90.py']:
 print('POSTSCORE90',name,flush=True)
 subprocess.run([sys.executable,str(H/name)],check=True)
