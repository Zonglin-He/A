"""Exact dependency-only overlay for the unchanged R50 text/engineering code.

No monkeypatch, dependency shadow of numpy/torch, model download or parser edit.
Import-smoke imports packages only, never constructs a Stanza pipeline/model.
The original target retains its own immutable plan/root authorization gates.
"""
from pathlib import Path
import sys,json,hashlib,argparse,time,runpy,datetime
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
OVERLAY=ROOT/'crossplatform/round50/text_runtime_overlay'
OVERLAY_SEAL_SHA256='7b4469834589c6fd7bf4091ee437399a546d84a9c13949a3b30a19831f8d1037'
TARGETS={
 'full_cpu':('crossplatform/round50/full_cpu50.py','b58553d187dee9960a093ba5357a4a03c76331fc1dce1e3dbebf43259a6afd8d'),
 'full_engineering':('crossplatform/round50/full_engineering50.py','95ce1d6f6fd20f40efffeb64c44cefe050cb3892fd95c5d1248072e4dd9f80f8')}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--import-smoke',action='store_true');p.add_argument('--target',choices=list(TARGETS));p.add_argument('--receipt')
 a,remaining=p.parse_known_args();sp=OVERLAY/'SEALED.json';assert sha(sp)==OVERLAY_SEAL_SHA256
 seal=json.loads(sp.read_text());assert seal['packages']=={'udapi':'0.5.2','colorama':'0.4.6','termcolor':'3.3.0'}
 actual={q.relative_to(OVERLAY).as_posix() for q in OVERLAY.rglob('*') if q.is_file() and q.name!='SEALED.json'}
 assert actual==set(seal['files'])
 for rel,h in seal['files'].items():assert sha(OVERLAY/rel)==h,rel
 sys.path.insert(0,str(OVERLAY));sys.path.insert(0,str(ROOT/'crossplatform/round20/runtime'))
 if a.import_smoke:
  assert a.receipt and not a.target and not remaining
  receipt=Path(a.receipt).resolve();assert receipt.is_relative_to(ROOT/'research/round50') and not receipt.exists()
  start=time.perf_counter();rec={'new_model_assets_loaded':False,'model_pipeline_constructed':False,'cohort_or_media_read':False,'overlay_sha256':OVERLAY_SEAL_SHA256}
  try:
   import stanza,udtools,udapi
   rec.update(state='PASS_R50_TEXT_DEPENDENCY_IMPORT_ONLY',modules={m.__name__:str(Path(m.__file__).resolve()) for m in (stanza,udtools,udapi)})
  except BaseException as e:
   rec.update(state='FAIL_R50_TEXT_DEPENDENCY_IMPORT_ONLY',error_type=type(e).__name__,error=str(e));raise
  finally:
   rec.update(seconds=time.perf_counter()-start,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());receipt.parent.mkdir(parents=True,exist_ok=True);receipt.write_text(json.dumps(rec,indent=2)+'\n')
  print(rec['state'],flush=True);return
 assert a.target and not a.receipt
 rel,digest=TARGETS[a.target];target=ROOT/rel;assert sha(target)==digest
 sys.argv=[str(target),*remaining];runpy.run_path(str(target),run_name='__main__')
if __name__=='__main__':main()
