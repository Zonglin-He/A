"""Cache-only original64 temporal mechanism audit, seal then evaluate."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,save,load,sha
BASE=ROOT/'artifacts/stvg_interval_decision_v1';OLD=ROOT/'artifacts/stvg_opd_paper_hc2_revision_v2/P6_existing_temporal_revision001/formal'
PUB=ROOT/'results/stvg_interval_decision/2026-10-18/cached64'
ARMS=['Native','projection','projection_length_native_location','shuffled_projection','native_MBR']
def prepare():
 files=[]
 for ds in ['hc2','vidstg']:
  pp=sorted((OLD/ds).glob('*.pt'));assert len(pp)==32
  for p in pp:
   r=read(p.with_suffix('.json'));assert sha(p)==r['sha256'];files.append(dict(path=str(p.relative_to(ROOT)),sha256=r['sha256'],dataset=ds,alias=('H'if ds=='hc2'else'V')+f'{len([x for x in files if x["dataset"]==ds])+1:03}'))
 put(BASE/'PLAN_LOCK.json',dict(status='locked_before_readouts',count=64,files=files,arms=ARMS,intervals='strict index s<e; original physical [fid_s,fid_e+1)',fusion='original physical envelope of two offsets',length_control='exact physical duration of projection, greatest Native legal posterior, lexicographic first tie',shuffle='one SHA256-seeded permutation per fixed source/offset; fixed values, physical weights unchanged',MBR='exact enumeration all legal intervals; physical continuous tIoU utility, first lexicographic maximum; no mass threshold',new_model_calls=0,GT_before_readout_seal=False,historical_development=True,created_epoch=time.time()))
 put(BASE/'RUNTIME_LOCK.json',dict(status='locked',pins={p:sha(ROOT/p)for p in ['vg_tta/stvg_interval_mbr_v1.py','scripts/run_stvg_interval_decision_v1.py','methods/decota_final_simplified_v1/objectives.py']},registries={p:sha(ROOT/p)for p in ['methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json']}))
def readout():
 import numpy as np,torch
 from types import SimpleNamespace
 from vg_tta.stvg_interval_mbr_v1 import legal_distribution,mbr
 from methods.decota_final_simplified_v1.objectives import project,native_view_indices
 torch.set_num_threads(4);files=[];checks=0
 for r in read(BASE/'PLAN_LOCK.json')['files']:
  assert sha(ROOT/r['path'])==r['sha256'];z=torch.load(ROOT/r['path'],map_location='cpu',weights_only=False);c=z['capture'];zero=c['zero'];cfg=SimpleNamespace(**z['config']);ev=project(zero['logits'],zero['actions'],c['records'],cfg);ivs={a:[]for a in ARMS};aud=[]
  actions=[]
  for off,(a,record)in enumerate(zip(zero['actions'],c['records'])):
   a=a.reshape(-1);seed=int(hashlib.sha256(('interval-shuffle-v1|'+r['alias']+'|'+str(off)).encode()).hexdigest()[:15],16);perm=np.random.default_rng(seed).permutation(len(a));actions.append(a[perm]);assert torch.equal(a.sort().values,actions[-1].sort().values)
  sh=project(zero['logits'],actions,c['records'],cfg)
  for off,(logits,record,t)in enumerate(zip(zero['logits'],c['records'],ev['offsets'])):
   iv,q,ij=legal_distribution(logits.numpy(),record['frame_ids']);ns,ne=native_view_indices(logits);native=[record['frame_ids'][ns],record['frame_ids'][ne]+1];pr=t['interval'];length=pr[1]-pr[0];pool=np.where(iv[:,1]-iv[:,0]==length)[0];assert len(pool)>0;at=int(pool[q[pool].argmax()]);mb=mbr(logits.numpy(),record['frame_ids'])
   values=[native,pr,iv[at].tolist(),sh['offsets'][off]['interval'],mb['interval']]
   for a,v in zip(ARMS,values):ivs[a].append(list(map(int,v)))
   x=t['raw_logits'];median=torch.quantile(x,.5);mad=torch.quantile((x-median).abs(),.5);u=t['standardized_logits'];w=t['omega'];pairs=t['ij'];lp=t['logp0'];prefix=torch.cat((u.new_zeros(1),(w*u).cumsum(0)));simple=(w*torch.nn.functional.logsigmoid(-u)).sum()+prefix[pairs[1]+1]-prefix[pairs[0]]+cfg.prior_weight*lp
   err=float((simple-t['score']).abs().max());assert err<1e-8
   shift=(x+3-cfg.center_fraction*(median+3))/(mad+cfg.epsilon)-u;expected=(1-cfg.center_fraction)*3/(mad+cfg.epsilon);assert float((shift-expected).abs().max())<1e-7
   aud.append(dict(offset=off,enumeration=len(q),projection_score_identity_max_error=err,additive_shift_identity_max_error=float((shift-expected).abs().max()),center_fraction=cfg.center_fraction,length_term_coefficient=float((1-cfg.center_fraction)*median/(mad+cfg.epsilon)),MBR=mb,exact_duration_control=True));checks+=5
  fused={a:[min(x[0]for x in v),max(x[1]for x in v)]for a,v in ivs.items()};assert fused['Native']==z['predictions']['Native']['physical_interval'];assert fused['projection']==z['predictions']['ActionnessProjection']['physical_interval'];checks+=2
  files.append(save(BASE/'readouts'/(r['alias']+'.pt'),dict(alias=r['alias'],dataset=r['dataset'],query_ordinal=z['query_ordinal'],frame_ids=z['frame_ids'],boxes=zero['boxes'],input_source_sha256=r['sha256'],offset_intervals=ivs,fused_intervals=fused,audit=aud,GT_read=False)))
 put(BASE/'GLOBAL_READOUT_BARRIER.json',dict(status='complete',count=64,files=files,GT_read=False,algebra_and_reference_checks=checks,new_model_calls=0))
 print('sealed64 temporal readouts',flush=True)
def score():
 import numpy as np
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_decota_paper_main_v1 import read_row
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
 p=read(BASE/'PLAN_LOCK.json');bar=read(BASE/'GLOBAL_READOUT_BARRIER.json');zs=[load(ROOT/r['path'])for r in bar['files']];targets={d:truths(d,dict(parents=[z['query_ordinal']for z in zs if z['dataset']==d]))for d in ['hc2','vidstg']};rows=[];checks=0
 for z in zs:
  ds=z['dataset'];q=z['query_ordinal'];gt=targets[ds][0][q];span=targets[ds][1][q];row=read_row(ds,q);tube=DenseTube(z['boxes'].numpy(),row,gt,span,clip=ds=='hc2');vals={}
  for support in ['offset0','offset1','fused']:
   vals[support]={}
   for arm in ARMS:
    iv=z['fused_intervals'][arm]if support=='fused'else z['offset_intervals'][arm][int(support[-1])];m=official(z['boxes'].numpy(),row,gt,span,iv,ds);assert max(abs(m[k]-tube.score(iv)[k])for k in ['v','t','s'])<2e-10;checks+=3
    m.update(length=float(iv[1]-iv[0]),R03=float(m['v']>.3),R05=float(m['v']>.5));vals[support][arm]=m
  rows.append(dict(source_alias=z['alias'],dataset=ds,values=vals))
 groups={d:{s:{a:{k:float(np.mean([r['values'][s][a][k]for r in rows if r['dataset']==d]))for k in ['v','t','s','R03','R05','length']}for a in ARMS}for s in ['offset0','offset1','fused']}for d in ['hc2','vidstg']}
 paired={}
 for d in ['hc2','vidstg']:
  paired[d]={}
  for support in ['offset0','offset1','fused']:
   paired[d][support]={}
   for a in ARMS[1:]:
    dif=np.array([r['values'][support][a]['v']-r['values'][support]['Native']['v']for r in rows if r['dataset']==d])*100;rg=np.random.default_rng(20261018);ci=np.quantile(dif[rg.integers(0,len(dif),(10000,len(dif)))].mean(1),[.025,.975]).tolist();paired[d][support][a]=dict(mean_pp=float(dif.mean()),CI95=ci,positive=int((dif>1e-9).sum()),negative=int((dif< -1e-9).sum()),neutral=int((abs(dif)<=1e-9).sum()),per_source_pp=dif.tolist())
 # Independently recompute scalar aggregate means from sealed per-source rows.
 for d in groups:
  for s in groups[d]:
   for a in groups[d][s]:
    for k,v in groups[d][s][a].items():assert abs(v-sum(r['values'][s][a][k]for r in rows if r['dataset']==d)/32)<1e-12;checks+=1
 PUB.mkdir(parents=True,exist_ok=True)
 put(PUB/'RESULTS.json',dict(status='complete',count=64,per_domain=32,groups=groups,paired=paired,rows=rows,historical_exposed=True,no_parameter_adaptation=True,shuffle_is_one_fixed_permutation_not_semantic_oracle=True))
 put(PUB/'VERIFICATION.json',dict(status='pass',count=64,independent_metric_and_mean_checks=checks,algebra_checks=bar['algebra_and_reference_checks'],all_predictions_sealed_before_new_GT_scoring=True,all_legal_intervals_enumerated=True,no_new_GPU_model_or_expert_calls=True))
 lines=['# Cached64 temporal decision audit','', 'Original32 exposed parents/domain; fixed Source boxes, original two offsets and physical envelope. Exact expected-tIoU MBR has no Actionness, extra model, threshold or parameter update. Historical Native/projection are replayed exactly.','', '| Domain | Arm (fused) | vIoU % | tIoU % | Δv vs Native pp |','|---|---|---:|---:|---:|']
 for d in groups:
  for a in ARMS:
   v=groups[d]['fused'][a];lines.append(f'| {d} | {a} | {100*v["v"]:.6f} | {100*v["t"]:.6f} | {100*(v["v"]-groups[d]["fused"]["Native"]["v"]):+.6f} |')
 lines+=['','Per-offset metrics, all signed source differences and paired parent bootstrap intervals are in RESULTS. Length is fixed exactly in physical frames before choosing location; this is a diagnostic, not an Actionness-free deployment variant because its length comes from projection. Shuffled values retain their multiset; one fixed permutation is a limited intervention, not a complete null distribution.','', 'MBR optimizes expected physical interval IoU under the Native model posterior, not unknown true IoU. Miscalibration or missing semantics can remain. Its decision rule is standard expected utility (https://aclanthology.org/2022.emnlp-main.754/), not claimed as new theory. No posterior-mass coverage threshold or GT-based choice is used.','', 'Production registries and prior negative findings stay unchanged.']
 (PUB/'REPORT.md').write_text('\n'.join(lines)+'\n');put(BASE/'CPU_COMPLETION.json',dict(status='complete',count=64,GT_read=True));put(BASE/'ROOT_READBACK.json',dict(status='pass',count=64,checks=checks));print(json.dumps(groups,indent=2))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','readout','score']);globals()[a.parse_args().stage]()
