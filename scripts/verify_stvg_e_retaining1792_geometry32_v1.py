"""Independent post-seal GT reference, full-tube and actual-frame validation."""
import os,sys,collections,time
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_e_retaining1792_route32_v1 import *

def main():
 import torch,numpy as np
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube
 torch.set_num_threads(4);p=verify(True);assert all(read(D/(s.upper()+'_PREDICTION_BARRIER.json'))['count']==32 for s in STAGES);res=read(P/'RESULTS_PUBLIC.json');g={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in g};c=collections.Counter()
 for r in p['records']:
  a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];row=plans[ds][q];gt,span=g[ds][0][q],g[ds][1][q];real=load(D/'inputs_real'/(a+'.pt'));oracle=load(D/'inputs_oracle'/(a+'.pt'));ft=load(FEAT/'features'/(a+'.pt'));src=load(PRE/'source'/(a+'.pt'));expected=[dict(position=f['position'],box=f['boxes'][k].tolist(),weight=1.)for f,k in zip(ft['frames'],ft['selected'])if k>=0];assert real['anchors']==expected;known=[v for v in expected if src['frame_ids'][v['position']]in gt];assert len(known)==len(oracle['anchors']);c['real_reference_reconstruction_GT_unfiltered']+=1
  for orig,o in zip(known,oracle['anchors']):
   assert orig['position']==o['position'];f=src['frame_ids'][o['position']];x,y,x2,y2=gt[f];W,H=row['input']['width'],row['input']['height'];assert np.array_equal(o['box'],[(x+x2)/(2*W),(y+y2)/(2*H),(x2-x)/W,(y2-y)/H]);c['GT_reference_exact_known_only']+=1
  for stage in STAGES:
   z=load(D/stage/(a+'.pt'));saved=next(t for t in res['rows']if t['alias']==a and t['stage']==stage);W,H=row['input']['width'],row['input']['height'];ids=z['frame_ids']
   def iou(b,f):
    b=np.asarray(b);xy=np.r_[b[:2]-b[2:]/2,b[:2]+b[2:]/2]*[W,H,W,H]
    if ds=='hc2':xy=np.maximum(xy,0)
    tr=np.asarray(gt[f]);inter=np.maximum(0,np.minimum(xy[2:],tr[2:])-np.maximum(xy[:2],tr[:2])).prod();area=lambda b:np.maximum(0,b[2:]-b[:2]).prod();return float(inter/max(area(xy)+area(tr)-inter,1e-12))
   for arm,b in z['outputs'].items():
    calc=DenseTube(b,row,gt,span,clip=ds=='hc2').score(z['interval'])
    for k in ['v','t','s']:assert abs(calc[k]-saved['values'][arm][k])<2e-10;c['independent_full_tube_metrics']+=1
    for name,mask in [('observed_actual_s',True),('unobserved_actual_s',False)]:
     vals=[iou(b[j],f)for j,f in enumerate(ids)if f in gt and((j in z['planned_positions'])==mask)];v=float(np.mean(vals))if vals else None;old=saved['values'][arm][name];assert v is None and old is None or v is not None and old is not None and abs(v-old)<1e-12;c['independent_actual_node_metrics']+=1
 assert not torch.cuda.is_initialized();verify(True);put(P/'SCIENTIFIC_READBACK_AUDIT.json',dict(status='pass',checks=dict(c),real_refs_not_GT_filtered=True,GT_used_only_oracle_and_postseal_metrics=True,unknown_support_not_fabricated=True,no_models_GPU=True));print('GEOMETRY PASS',dict(c),flush=True)
if __name__=='__main__':main()
