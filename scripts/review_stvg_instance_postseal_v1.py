"""Independent complete-scope seal and score verification; no inference or updates."""
import sys,math,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_instance_identity_common_v1 import BASE,OLD,read,write,sha

def need(ok,message):
 if not ok:raise RuntimeError(message)
def main():
 import torch,numpy as np
 from scripts.stvg_sparse_r1_prompt_v1 import load_source_case
 from scripts import score_stvg_native_instance_nativeclock005 as m
 from scripts import review_stvg_sparse_r1_v2 as sc
 roster=read(OLD/'ROSTER_LOCK.json')['records'];ids=[r['case_id']for r in roster]
 need(len(set(ids))==8 and collections.Counter(r['dataset']for r in roster)=={'hc2':4,'vidstg':4},'roster')
 base_files=['DESIGN_LOCK.json','P1_UPDATE_LOCK.json','STUDENT_RUNTIME.json','STUDENT_GLOBAL_BARRIER.json','REFERENCE_LOCK.json','TEACHER_GEOMETRY_ONLY_READOUT.json','ROOT_INDEPENDENT_GEOMETRY_READBACK.json','STEP_DIAGNOSTIC_READBACK.json']
 expected={str((BASE/n).relative_to(ROOT))for n in base_files}
 expected|={str((BASE/'student'/(cid+suffix)).relative_to(ROOT))for cid in ids for suffix in ['.pt','.receipt.json']}
 seal=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');need(seal['status']=='complete'and seal['count']==8 and seal['GT_read']is False,'seal state')
 need(len(seal['files'])==24 and {x['path']for x in seal['files']}==expected,'complete unique 24-file seal')
 for x in seal['files']:
  p=ROOT/x['path'];need(p.resolve().is_relative_to(ROOT)and p.stat().st_size==x['bytes']and sha(p)==x['sha256'],'sealed bytes')
 for p,h in read(BASE/'STUDENT_RUNTIME.json')['pins'].items():need(sha(ROOT/p)==h,'source pin')
 need(seal['scorer_sha256']==sha(ROOT/'scripts/score_stvg_instance_identity_v1.py'),'scorer pin')
 refdoc=read(BASE/'REFERENCE_LOCK.json');geodoc=read(BASE/'TEACHER_GEOMETRY_ONLY_READOUT.json')
 need(refdoc['status']=='locked'and geodoc['status']=='complete'and refdoc['count']==geodoc['count']==8,'reference status')
 refs={x['case_id']:x for x in refdoc['records']};geo={x['case_id']:x for x in geodoc['records']};need(set(refs)==set(geo)==set(ids),'reference scope')
 arms={'F','T'}|{s+n for n in ['strict_iou','raw_iou','supported_iou','loss_weighted_iou','validated_loss_weighted_iou']for s in ['','T_']}
 cases={};data={};redecode_count=0
 for r in roster:
  cid=r['case_id'];need(sha(ROOT/r['source_case_path'])==r['source_case_sha256'],'source case')
  c=load_source_case(ROOT/r['source_case_path']);d=torch.load(BASE/'student'/(cid+'.pt'),map_location='cpu',weights_only=False)
  need(d['case_id']==cid and d['dataset']==c['dataset']==r['dataset']and set(d['arms'])==arms,'case dataset or arm set')
  need(list(c['frame_ids'])==d['frame_ids'],'frame grid')
  for a in refs[cid]['raw_decoupled']['anchors']:
   p=a['position'];need(type(p)is int and 0<=p<len(d['frame_ids'])and d['frame_ids'][p]==a['physical_frame_id'],'support map')
  for name,f in d['fits'].items():
   need(torch.equal(f['final_cached_decoder_boxes'].cpu(),d['readouts'][name]['boxes'].cpu()),'complete final decoder');redecode_count+=1
  for n,k in [('T_geometry_strict','strict_original'),('T_geometry_raw','raw_decoupled')]:
   g=geo[cid][k];need(g['frame_ids_physical']==d['frame_ids']and g['interval_frame_bounds_half_open']==d['arms']['T']['interval'],'geometry grid/time')
   d['arms'][n]=dict(boxes=g['boxes_xyxy_normalized'],interval=g['interval_frame_bounds_half_open'])
  cases[cid]=c;data[cid]=d
 # Only after complete input verification, repeat the official/independent post-seal evaluation.
 targets,_=sc._truths_after_seal(cases);official,independent,_=sc._metric_functions()
 rows=read(BASE/'POSTSEAL_ROWS.json')['rows'];need(len(rows)==112,'score row count');lookup={(r['case_id'],r['arm']):r for r in rows};need(len(lookup)==112,'score unique')
 checks=0
 for cid,d in data.items():
  support={a['position']for a in refs[cid]['raw_decoupled']['anchors']}
  for name,a in d['arms'].items():
   mm,audit=m._score_student_arm(a,cases[cid],targets[cid],support,official,independent)
   saved=lookup[cid,name];need(saved['dataset']==d['dataset'],'scored dataset')
   for k,v in mm.items():need((v is None and saved['metrics'][k] is None)or(v is not None and abs(v-saved['metrics'][k])<1e-12),'score readback');checks+=1
 summary=read(BASE/'POSTSEAL_SUMMARY.json')
 for s in summary['summary']:
  v=[r['metrics'][s['metric']]for r in rows if r['dataset']==s['dataset']and r['arm']==s['arm']and r['metrics'][s['metric']]is not None]
  need(s['count']==len(v)and s['unknown_count']==4-len(v),'count');need(s['mean']is None if not v else abs(s['mean']-math.fsum(v)/len(v))<1e-12,'mean')
 for c in summary['contrasts']:
  diffs=[]
  for cid in ids:
   if data[cid]['dataset']!=c['dataset']:continue
   a=lookup[cid,c['left']]['metrics'][c['metric']];b=lookup[cid,c['right']]['metrics'][c['metric']]
   if a is not None and b is not None:diffs.append(a-b)
  need(c['paired_bootstrap_95_pp']==m._paired_bootstrap(diffs,c['dataset'],c['left']+'-'+c['right'],c['metric']),'bootstrap')
 write(BASE/'ROOT_POSTSEAL_READBACK.json',dict(status='pass',GT_read=True,case_count=8,sealed_files=24,score_rows=112,metric_checks=checks,summary_cells=len(summary['summary']),paired_contrasts=len(summary['contrasts']),full_final_readbacks= redecode_count,full_actual_scope_preflight=True,scorer_general_hardening='original scorer uses assert and weaker generic gates; this independent verifier validates actual fixed full scope using explicit exceptions',scorer_sha256=sha(ROOT/'scripts/score_stvg_instance_identity_v1.py'),reviewer_sha256=sha(__file__)))
 print('pass',checks)
if __name__=='__main__':main()
