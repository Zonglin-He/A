"""Independent metadata/serialization audit before any task-GT scoring."""
from pathlib import Path
import json,hashlib,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round128';B=R/'crossplatform/round128';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 seal=rd(H/'FAST_MATRIX_SEALED.json');n=0;aliases=0;maxerr=0;stats={k:[] for k in ['tent','memo','eata']}
 for name,sh in seal['seals'].items():
  D=B/'formal_fast'/name;assert sha(D/'SEALED.json')==sh;ss=rd(D/'SEALED.json');assert ss['n']==128
  for p,h in ss['files'].items():assert sha(D/p)==h
  for f in sorted((D/'samples_PRIVATE').glob('*.npz')):
   a=np.load(f,allow_pickle=False);r=rd(D/'records_PRIVATE'/(f.stem+'.json'));assert a['scores'].shape==(12,256);assert np.isfinite(a['scores']).all() and np.allclose(a['scores'].sum(1),1,atol=1e-10);assert np.array_equal(a['scores'].argmax(1),a['queries']);n+=1
   if r['protected_alias_fallback']:aliases+=1;assert len(set(a['queries']))==1;continue
   maxerr=max(maxerr,r['source_replay_max_probability_error']);assert maxerr<2e-5
   for k in stats:
    q=r[k];steps=q['steps'];accepted=sum(t['accepted'] for t in steps);assert accepted==0 or (q['parameter_delta_L2']>0 and any(t['gradient_L2']>0 for t in steps));stats[k].append({'accepted_steps':accepted,'parameter_delta_L2':q['parameter_delta_L2'],'selected_parameters':max(t['selected_parameters'] for t in steps),'gradient_parameters':max(t['gradient_parameters'] for t in steps),'seconds':q['seconds'],'peak_GPU_GiB':q['peak_GPU_GiB'],'Fisher_max_penalty':max(t.get('Fisher_penalty',0.) for t in steps)})
 assert n==768 and aliases==44;out={'status':'PASS_NATIVE_FULL_MODEL_PORTS_SEALED','n':n,'actual_neural_inputs':n-aliases,'protected_alias_fallbacks':aliases,'source_replay_max_probability_error':maxerr,'methods':{k:{'actual_updates':sum(s['accepted_steps'] for s in a),'inputs_updated':sum(s['accepted_steps']>0 for s in a),'multi_step_inputs':sum(s['accepted_steps']>1 for s in a),'nonzero_Fisher_penalty_inputs':sum(s['Fisher_max_penalty']>0 for s in a),'selected_parameters':max(s['selected_parameters'] for s in a),'max_gradient_parameters':max(s['gradient_parameters'] for s in a),'mean_seconds':float(np.mean([s['seconds'] for s in a])),'max_GPU_GiB':max(s['peak_GPU_GiB'] for s in a)} for k,a in stats.items()},'task_GT_read':False};(H/'FAST_NATIVE_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
