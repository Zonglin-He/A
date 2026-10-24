"""Explicit stdlib import adapter; frozen structural science and outputs unchanged."""
from common import *
import argparse,shutil
import spatial as frozen

def prepare():
 p=freeze('spatial_runtime_v2',dict(parent_spatial_plan_sha256=sha(D/'spatial_plan.json'),change='Supply missing stdlib shutil global used only for disk capacity guard before first query. Original predict/derive/score/audit function objects and science unchanged.',preservation='Caught statically before any spatial production work or queuedGPU. Old queue waiting owner safely stopped; no paid work lost; no old frozen hash modified.'),[Path(__file__),S/'common.py',S/'spatial.py',S/'spatial_core.py']);print(p['fingerprint'])
def run(command,ds=None):
 validate('spatial_runtime_v2');frozen.validate();assert 'shutil'not in vars(frozen);frozen.shutil=shutil
 if command=='preflight':return frozen.preflight()
 return dict(predict=frozen.predict,score=frozen.score,audit=frozen.audit)[command](ds)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','preflight','predict','score','audit']);a.add_argument('--dataset',choices=COUNTS);z=a.parse_args();prepare()if z.command=='prepare'else run(z.command,z.dataset)
