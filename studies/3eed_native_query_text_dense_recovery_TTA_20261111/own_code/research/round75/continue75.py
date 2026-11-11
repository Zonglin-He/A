"""Sequential guarded stages; never run concurrent GPU jobs."""
from common75 import *
import subprocess
for resource,prog,args in [('cpu','fit75.py',[]),('gpu','capture75.py',['--held']),('cpu','predict75.py',[])]:
 subprocess.run([sys.executable,str(ROOT/'tools/manual_r75_guard.py'),'--resource',resource,'--timeout-seconds','1800','--',str(ROOT/'crossplatform/env/bin/python'),str(H/prog)]+args,check=True)
