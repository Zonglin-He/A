from pathlib import Path
R=Path(__file__).resolve().parents[2]
for source,target in [('parse23.py','parseM24.py'),('prepare23.py','prepareM24.py'),('detect23.py','detectM24.py'),('evidence23.py','evidenceM24.py'),('rank23.py','rankM24.py')]:
 s=(R/'crossplatform/round23'/source).read_text().replace('from common23 import *','from common24 import *').replace("'K'","'M'").replace('ALL272_PRIVATE','M64_PRIVATE').replace('text272_attempt001','textM64_attempt001').replace('prepared272_attempt001','preparedM64_attempt001').replace('dino272_attempt001','dinoM64_attempt001').replace('evidence272_attempt001','evidenceM64_attempt001').replace('bank272_attempt001','bankM64_attempt001').replace('sourceK64_attempt001','capture336_attempt001').replace('n=272','n=64')
 (R/'crossplatform/round24'/target).write_text(s)
import sys
sys.path.insert(0,str(R/'crossplatform/round24'))
from common24 import dump,sha,EXP
files=list((R/'crossplatform/round24').glob('*.py'))+[R/'research/round24/select24.py',Path(__file__)]
dump(EXP/'CODE_LOCK.json',{'programs':{str(p.relative_to(R)):sha(p) for p in files}})
