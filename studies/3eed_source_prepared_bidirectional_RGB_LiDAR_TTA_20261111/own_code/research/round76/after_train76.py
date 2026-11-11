"""Bounded sequential continuation, original user scope only."""
from common76 import *
import subprocess
for prog in ['audit_source76.py','audit_masking76.py','target76.py']:
 subprocess.run([sys.executable,str(ROOT/'tools/manual_r76_guard.py'),'--resource','gpu','--timeout-seconds','3600','--',str(ROOT/'crossplatform/env/bin/python'),str(H/prog)],check=True)
