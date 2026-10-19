"""Lossless public scalar sharding; original full steps preserved privately.

No predictions, scores, seeds or scientific configuration are changed. The
public index plus all shards reconstructs every original per-fit step scalar.
"""
from pathlib import Path
import json,hashlib,shutil,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,sha
from scripts.run_stvg_referdino_native_opd_hc128_v1 import D,P

def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def main():
 assert read(P/'ROOT_INDEPENDENT_AUDIT.json')['status']=='pass';assert read(P/'READOUT_AND_NATIVE_LABEL_AUDIT.json')['status']=='pass';rawdir=D/'publication_raw_scalars';rawdir.mkdir(exist_ok=True);source=P/'STEP_DIAGNOSTICS_PUBLIC.json';full=read(source);assert 'rows'in full and len(full['rows'])==896;shutil.copyfile(source,rawdir/'STEP_DIAGNOSTICS_FULL_ORIGINAL.json');shards=[];rebuilt=[]
 for i in range(8):
  rows=full['rows'][112*i:112*(i+1)];assert len(rows)==112;obj=dict(schema_version=1,rows=rows);name=f'STEP_DIAGNOSTICS_PART_{i:02}.json';f=P/name;assert not f.exists();f.write_text(json.dumps(obj,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n');rr=read(f)['rows'];rebuilt+=rr;shards.append(dict(path=name,fits=len(rr),steps=sum(len(r['steps'])for r in rr),bytes=f.stat().st_size,sha256=sha(f)))
 assert rebuilt==full['rows'];h=hashlib.sha256(canonical(full['rows'])).hexdigest();index={k:v for k,v in full.items()if k!='rows'};index.update(format='lossless JSON shards; concatenate rows in shard order',total_fits=896,total_steps=sum(len(r['steps'])for r in rebuilt),logical_rows_canonical_SHA256=h,shards=shards);assert index['total_steps']==35840;source.write_text(json.dumps(index,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 packed=[]
 for f in P.glob('*.json'):
  if f.stat().st_size>512000 and not f.name.startswith('STEP_DIAGNOSTICS_PART_'):
   obj=read(f);shutil.copyfile(f,rawdir/f.name);f.write_text(json.dumps(obj,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n');assert read(f)==obj;packed.append(f.name)
 put(P/'SCALAR_EXPORT_AUDIT.json',dict(status='pass',all896_fit35840_step_scalars_preserved=True,logical_rows_canonical_SHA256=h,shards=shards,JSON_whitespace_only_minification=packed,no_metric_prediction_or_configuration_change=True,private_pretty_copies_preserved=True))
 fl=read(D/'PUBLIC_FILES_FINAL.json')['files'];fl+=[str((P/x['path']).relative_to(ROOT))for x in shards]+[str((P/'SCALAR_EXPORT_AUDIT.json').relative_to(ROOT)),str((P/'READOUT_AND_NATIVE_LABEL_AUDIT.json').relative_to(ROOT))];fl=list(dict.fromkeys(fl));put(D/'PUBLIC_FILES_FINAL.json',dict(files=fl),True);print('LOSSLESS PUBLIC SCALARS',len(rebuilt),index['total_steps'],len(shards),flush=True)
if __name__=='__main__':main()
