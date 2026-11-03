"""Combine the locked11 arms and bounded20step direct controls, no GT."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan)
 for rel,h in {**p['programs_sha256'],**p['inputs_sha256']}.items():assert sha(ROOT/rel)==h
 rows=read(ROOT/p['manifest']);main=ROOT/p['main_root'];strong=ROOT/p['strong_root'];ms=read(main/'SEALED.json');ss=read(strong/'SEALED.json');arms=ms['arm_names']+ss['arm_names'];out=ROOT/p['output_root'];assert not out.exists();out.mkdir(parents=True);(out/'samples_PRIVATE').mkdir();status=[]
 sm={r['id']:r['arm_statuses'] for r in read(main/'READOUT_STATUS_PRIVATE.json')};st={r['id']:r['arm_statuses'] for r in read(strong/'READOUT_STATUS_PRIVATE.json')}
 for row in rows:
  sid=row['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(main/rel)==ms['files'][rel] and sha(strong/rel)==ss['files'][rel]
  with np.load(main/rel,allow_pickle=False) as m,np.load(strong/rel,allow_pickle=False) as s:
   assert np.array_equal(m['source_boxes'],s['source_boxes']) and np.array_equal(m['source_scores'],s['source_scores']);scores=np.concatenate([m['scores'],s['scores']],0);q=scores.argmax(1).astype(np.int64);assert np.array_equal(q,np.r_[m['queries'],s['queries']]);np.savez_compressed(out/rel,arm_names=np.asarray(arms),scores=scores,queries=q,source_boxes=m['source_boxes'],source_scores=m['source_scores'],selected_original_boxes=m['source_boxes'][q])
  status.append({'id':sid,'arm_statuses':{**sm[sid],**st[sid]}})
 dump(out/'READOUT_STATUS_PRIVATE.json',status);dump(out/'SEALED.json',{'state':'R54_ALL13_FIXED_SAME_INFORMATION_READOUTS_NO_GT','n':len(rows),'arm_names':arms,'plan_sha256':sha(a.plan),'main_seal_sha256':sha(main/'SEALED.json'),'strong_control_seal_sha256':sha(strong/'SEALED.json'),'files':{f.relative_to(out).as_posix():sha(f) for f in sorted(out.rglob('*')) if f.is_file()}});print(json.dumps({'state':'SEALED','n':len(rows),'arms':len(arms)}),flush=True)
if __name__=='__main__':main()
