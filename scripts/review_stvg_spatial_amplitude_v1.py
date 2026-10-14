"""Independent CPU state, endpoint and official/dense metric readback after full seal."""
import sys,json,hashlib,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from scripts.review_stvg_sparse_r1_v2 import _truths_after_seal,_metric_functions,score_tube_xyxy
from scripts.stvg_sparse_r1_prompt_v1 import load_source_case
from scripts.score_stvg_motion_guided_postseal_v1 import _per_frame_iou,_subset_delta
from methods.decota_final_simplified_v1.tensors import state_hash
BASE=ROOT/'artifacts/stvg_spatial_causal_decomposition_v1';P=BASE/'phase1';OLD=ROOT/'artifacts/stvg_motion_guided_instance_opd_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def main():
 bar=read(P/'PHASE1_GLOBAL_BARRIER.json');assert bar['status']=='complete' and bar['count']==16 and not bar['GT_read'];assert bar['phase1_runtime_sha256']==sha(P/'RUNTIME.json')
 runtime=read(P/'RUNTIME.json');
 for rel,d in {**runtime['pins'],**runtime['protected']}.items():assert sha(ROOT/rel)==d
 assert len(bar['files'])==16 and len({f['path']for f in bar['files']})==16
 outputs={}
 for f in bar['files']:
  p=ROOT/f['path'];assert sha(p)==f['sha256'] and p.stat().st_size==f['bytes'];o=torch.load(p,map_location='cpu',weights_only=False);assert not o['GT_read'];assert o['case_id'] not in outputs;assert set(o['amplitudes'])=={'0','0.25','0.5','1'};outputs[o['case_id']]=o
 roster=read(OLD/'ROSTER_LOCK.json')['records'];assert set(outputs)=={r['case_id']for r in roster};cases={};bindings=[]
 oldscores=read(OLD/'ROOT_POSTSEAL_REPORT.json')['metric_rows'];oldmetrics={(r['case_id'],r['arm']):r['metrics']for r in oldscores if r['phase']=='method'}
 for rec in roster:
  cid=rec['case_id'];o=outputs[cid];oldpath=OLD/'method/student'/f'{cid}.pt';assert sha(oldpath)==o['old_method_output_sha256'];old=torch.load(oldpath,map_location='cpu',weights_only=False)
  p=ROOT/rec['source_case_path'];assert sha(p)==rec['source_case_sha256'];cases[cid]=load_source_case(p);assert o['source_case_sha256']==rec['source_case_sha256'];fit=old['fits'].get('teacher_iou')
  for k,a in o['amplitudes'].items():
   v=float(k)
   if a['full_model_redecode']:
    expected={n:(x.clone() if v==0 else fit['final_state'][n].clone() if v==1 else x+(fit['final_state'][n]-x)*v)for n,x in fit['initial_state'].items()};assert state_hash(expected)==a['state_sha256'];assert a['random_state_unchanged'] and a['source_model_restored']
    if v in [0,1]:assert torch.equal(a['boxes_cxcywh'],old['source_boxes'] if v==0 else old['readouts']['teacher_iou']['boxes'])
   else:assert a['exact_alias_of']=='source_readout';assert torch.equal(o['source_readout']['boxes_cxcywh'],old['source_boxes'])
  bindings.append(dict(case_id=cid,source_sha256=rec['source_case_sha256'],old_output_sha256=sha(oldpath),state_and_endpoints_exact=True))
 # Only now expose target annotations for metrics; never used in new state or reference selection.
 truth,provenance=_truths_after_seal(cases);official,independent,_=_metric_functions();rows=[]
 refs={r['case_id']:r['references']['teacher']for r in read(OLD/'METHOD_REFERENCE_LOCK.json')['records']}
 for rec in roster:
  cid=rec['case_id'];o=outputs[cid];c=cases[cid];fid=list(map(int,c['frame_ids']));supervised={fid[x['position']]for x in refs[cid].get('anchors',[])}
  for k,a in o['amplitudes'].items():
   value=a if a['full_model_redecode'] else o['source_readout'];boxes=np.asarray(value['boxes_xyxy'],float);intervals={'S':o['source_readout']['physical_interval']}
   if o['temporal_reuse']['interval']is not None:intervals['TS']=o['temporal_reuse']['interval']
   for branch,interval in intervals.items():
    q=score_tube_xyxy(boxes,fid,c['width'],c['height'],interval,truth[cid]['truth'],truth[cid]['span'],c['dataset'],official=official,independent=independent)
    expectedarm=('F' if branch=='S' else 'T')if k=='0' else ('teacher_iou' if branch=='S' else 'T_teacher_iou')
    if k in ['0','1']:assert max(abs(q['metrics'][m]-oldmetrics[(cid,expectedarm)][m])for m in q['metrics'])<2e-9
    frame=_per_frame_iou(dict(boxes=boxes,interval=interval),fid,c['width'],c['height'],c['dataset'],truth[cid]['truth'],truth[cid]['span'])
    baseline=_per_frame_iou(dict(boxes=np.asarray(o['source_readout']['boxes_xyxy']),interval=interval),fid,c['width'],c['height'],c['dataset'],truth[cid]['truth'],truth[cid]['span'])
    observed=_subset_delta(baseline,frame,supervised,selection_status='fixed_reference_observations_GT_known_only')
    unseen=_subset_delta(baseline,frame,set(map(int,baseline.fids))-supervised,selection_status='other_GT_known_frames_dense_geometry')
    rows.append(dict(case_id=cid,dataset=c['dataset'],cohort='original8'if int(cid[-3:])<4 else'additional8',alpha=float(k),branch=branch,metrics=q['metrics'],official_independent_max_abs_error=q['official_independent_max_abs_error'],baseline_delta_vIoU_pp=100*(q['metrics']['m_vIoU']-oldmetrics[(cid,'F' if branch=='S'else'T')]['m_vIoU']),fresh_model_readout=a['full_model_redecode'],supervised_physical_frame_count=len(supervised),GT_support_count=len(truth[cid]['truth']),supervised_geometry=observed,unsupervised_geometry=unseen))
 groups=[]
 for cohort in ['original8','additional8']:
  for dataset in ['hc2','vidstg']:
   for branch in ['S','TS']:
    for alpha in [0,.25,.5,1]:
     rr=[r for r in rows if(r['cohort'],r['dataset'],r['branch'],r['alpha'])==(cohort,dataset,branch,alpha)]
     if not rr:continue
     groups.append(dict(cohort=cohort,dataset=dataset,branch=branch,alpha=alpha,count=len(rr),mean_vIoU=statistics.mean(r['metrics']['m_vIoU']for r in rr),mean_delta_vIoU_pp=statistics.mean(r['baseline_delta_vIoU_pp']for r in rr),positive=sum(r['baseline_delta_vIoU_pp']>1e-8 for r in rr),negative=sum(r['baseline_delta_vIoU_pp']< -1e-8 for r in rr),neutral=sum(abs(r['baseline_delta_vIoU_pp'])<=1e-8 for r in rr)))
 out=dict(status='pass',scope='finite fixed saved update amplitude; no selected alpha or production promotion',barrier_sha256=sha(P/'PHASE1_GLOBAL_BARRIER.json'),case_count=16,metric_count=len(rows),actual_new_readouts=bar['actual_full_model_redecode_calls'],new_fits=0,GT_read=True,GT_used_for_method_selection=False,source_bindings=bindings,GT_source_bindings=provenance,rows=rows,groups=groups,max_official_independent_error=max(r['official_independent_max_abs_error']for r in rows))
 dest=P/'ROOT_AMPLITUDE_READBACK.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(status='pass',metric_count=len(rows),groups=groups),indent=2))
if __name__=='__main__':main()
