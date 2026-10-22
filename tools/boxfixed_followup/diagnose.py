"""Offline diagnosis of sealed fixed128 evidence; no model or new predictions."""
import json,sys,hashlib,collections
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.boxfixed_tube.profiles import view_geometry
from tools.boxfixed_tube.spatial import build_geometry,BoxFixedBank,evidence_dp
from stvg.predictor import output_writeback
from stvg.metrics_base import xyxy
from stvg.metrics import box_iou
P=ROOT/'results/tubedetr_boxfixed_development';OUT=ROOT/'outputs/tubedetr_boxfixed_followup';PUBLIC=ROOT/'results/tubedetr_boxfixed_followup'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,a):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
def avg(x):x=[float(a)for a in x if a is not None and np.isfinite(a)];return float(np.mean(x))if x else None
def summaries(rows):
 keys=['gt_cells','gt_width_cells','gt_height_cells','anchor_seconds','source_node_iou','E_node_iou','selected_node_iou','oracle_node_iou','prop_node_iou','prop_wins_fraction','selected_harm_fraction','source_oracle_gap','mean_confidence','cos_background','adjacent_same_location_cos','mean_fit_scale','seed_self_dice','valid_fit_fraction','Full_minus_Temporal_v_pp','Full_minus_Temporal_s_pp']
 return dict(queries=len(rows),**{k:avg([r.get(k)for r in rows])for k in keys})
def main():
 torch.set_num_threads(1);plan=read(P/'plan.json');output=read(PUBLIC/'diagnosis.json')['datasets'] if '--hc-only' in sys.argv else {}
 for ds in (['hc2'] if '--hc-only' in sys.argv else ['vidstg','hc2']):
  d=plan['datasets'][ds];manifest=ROOT/d['manifest'];assert h(manifest)==d['manifest_sha256'];requests=[json.loads(x)for x in manifest.read_text().splitlines()];root=ROOT/'outputs/tubedetr_boxfixed_development'/ds/'captures';seal=read(root/'sealed.json');selection=read(P/ds/'selected_profile.json');params=selection['parameters'];assert seal['queries']==128 and seal['plan_sha256']==h(P/'plan.json')
  selected=ROOT/'outputs/tubedetr_boxfixed_development'/ds/'profiles'/selection['profile_id'];ps=read(selected/'sealed.json');assert h(selected/'sealed.json')==selection['dev_prediction_seal_sha256']
  for s,r in [(seal,root),(ps,selected)]:
   for e in s['files'].values():assert h(r/e['path'])==e['sha256'] and h(r/e['receipt_path'])==e['receipt_sha256']
  # All old captures and selected predictions are verified before label access.
  assert h(ROOT/d['labels'])==d['labels_sha256'];labels={a['id']:a for a in map(json.loads,(ROOT/d['labels']).read_text().splitlines())};details={a['id']:a for a in read(selected/'details_private.json')};rows=[];frame_rows=[];fit_reasons=collections.Counter()
  for qi,request in enumerate(requests):
   q=request['id'];c=torch.load(root/seal['files'][q]['path'],weights_only=False,map_location='cpu');ev=c['evidence'];S=c['Source'];T=len(S);E=torch.from_numpy(output_writeback(S,c['raw'],params['gamma']));g0=build_geometry(ev,E,c['native_boxes']);g=view_geometry(g0,ev,S,c['raw'],params);bank=BoxFixedBank(g,interval=c['Global6_interval']);
   # Exact selected replay, including its tuned score weights.
   from tools.boxfixed_tube.profiles import predict
   pred=predict(g0,ev,S,c['raw'],c['Global6_interval'],params)
   with np.load(selected/ps['files'][q]['path'])as z:assert np.array_equal(pred['final'].numpy(),z['BoxFixed'])
   gt=labels[q];truth={int(k):v for k,v in gt['boxes'].items()};nodes=[t for t,f in enumerate(c['frame_ids'])if int(f)in truth];idx=np.asarray(nodes);ids=np.asarray(c['frame_ids']);target=np.array([truth[int(ids[t])]for t in nodes]);meta=request['metadata'];w,hh=meta['width'],meta['height'];source=xyxy(S,w,hh);cand=xyxy(g.boxes.numpy().reshape(-1,4),w,hh).reshape(T,3,4);source=np.maximum(source,0) if ds=='hc2' else source;cand=np.maximum(cand,0) if ds=='hc2' else cand;ious=box_iou(cand[idx],target[:,None]);siou=box_iou(source[idx],target);active=bank.active.numpy()[idx];valid=bank.valid.numpy()[idx].copy();valid[~active,1:]=False;oracle=np.max(np.where(valid,ious,-1),axis=1);piou=np.max(np.where(valid[:,1:],ious[:,1:],-1),axis=1);actions=pred['actions'].numpy()[idx];actual=ious[np.arange(len(idx)),actions];obs=g.observed.numpy()[idx];event=bank.event.numpy()[idx];inside=(ids[idx]>=gt['interval'][0])&(ids[idx]<gt['interval'][1]);take=event&inside;unobs=take&~obs;obstake=take&obs
   area=np.maximum(target[:,2:]-target[:,:2],0);cells=area[:,0]*area[:,1]/(w*hh)*ev.height*ev.width;dist=np.array([min(abs(int(ids[t])-int(ids[a]))for a in ev.positions)/meta['fps']for t in nodes]);F=torch.nn.functional.normalize(ev.features,dim=-1);cos=(F[ev.positions]@F[ev.positions].transpose(-1,-2)).numpy();off=~np.eye(cos.shape[-1],dtype=bool);scales=[];validfits=[]
   for t in range(T):
    if g.observed[t]:continue
    for side in (0,1):
     fit=g.fit_diagnostics[t][side]
     if fit is None:continue
     fit_reasons[fit['reason']]+=1;validfits.append(fit['valid'])
     if fit['valid']:scales.extend(fit['scale'])
   selfdice=((ev.seeds.square().sum((-2,-1)))/(ev.seeds.sum((-2,-1)).clamp_min(1e-15))).numpy();drow=details[q];r=dict(id=q,kind=meta.get('kind','human'),gt_cells=avg(cells),gt_width_cells=avg(area[:,0]/w*ev.width),gt_height_cells=avg(area[:,1]/hh*ev.height),anchor_seconds=avg(dist[unobs]),source_node_iou=avg(siou[unobs]),E_node_iou=avg(ious[unobs,0]),selected_node_iou=avg(actual[unobs]),oracle_node_iou=avg(oracle[unobs]),prop_node_iou=avg(piou[unobs][piou[unobs]>=0]),prop_wins_fraction=avg((piou[unobs]>siou[unobs])[piou[unobs]>=0]),selected_harm_fraction=avg((actual[unobs]-siou[unobs]<-.05)),source_oracle_gap=avg(np.maximum(oracle[unobs],siou[unobs])-siou[unobs]),mean_confidence=avg(g.confidence.numpy()[idx][unobs]),cos_background=float(cos[:,off].mean()),adjacent_same_location_cos=float((F[1:]*F[:-1]).sum(-1).mean()),mean_fit_scale=avg(scales),seed_self_dice=avg(selfdice),valid_fit_fraction=avg(validfits),Full_minus_Temporal_v_pp=100*(drow['values']['Full']['v']-drow['values']['Temporal']['v']),Full_minus_Temporal_s_pp=100*(drow['values']['Full']['s']-drow['values']['Temporal']['s']),observation_source_iou=avg(siou[obstake]),observation_selected_iou=avg(actual[obstake]),observations_in_GT=int(inside[obs].sum()),grid=[ev.height,ev.width]);rows.append(r)
   for n,t in enumerate(idx):
    if not unobs[n]:continue
    frame_rows.append(dict(cells=float(cells[n]),distance_seconds=float(dist[n]),confidence=float(g.confidence[t]),admitted=bool(active[n]),selected_action=int(actions[n]),source_iou=float(siou[n]),selected_iou=float(actual[n]),oracle_iou=float(oracle[n]),prop_iou=float(piou[n]),source_size=S[t,2:].tolist()))
   write(OUT/'diagnostic_progress.json',dict(dataset=ds,complete=qi+1,total=128));print(ds,qi+1,flush=True)
  def fsum(a):return dict(frames=len(a),source_iou=avg([x['source_iou']for x in a]),selected_iou=avg([x['selected_iou']for x in a]),oracle_iou=avg([x['oracle_iou']for x in a]),admitted_fraction=avg([x['admitted']for x in a]),selected_fraction=avg([x['selected_action']>0 for x in a]),confidence=avg([x['confidence']for x in a]))
  group={}
  for name,fn in [('small_lt3cells',lambda r:r['gt_cells']<3),('large_ge3cells',lambda r:r['gt_cells']>=3),('no_observation_inside_GT',lambda r:r['observations_in_GT']==0),('some_observation_inside_GT',lambda r:r['observations_in_GT']>0),('questions',lambda r:r['kind']=='questions'),('captions',lambda r:r['kind']=='captions')]:group[name]=summaries([r for r in rows if fn(r)])
  frames={name:fsum([r for r in frame_rows if fn(r)])for name,fn in [('all',lambda r:True),('small_lt3cells',lambda r:r['cells']<3),('large_ge3cells',lambda r:r['cells']>=3),('near_le1second',lambda r:r['distance_seconds']<=1),('far_gt1second',lambda r:r['distance_seconds']>1),('changed',lambda r:r['selected_action']>0),('admitted',lambda r:r['admitted'])]}
  output[ds]=dict(all=summaries(rows),groups=group,frame_groups=frames,fit_reasons=dict(fit_reasons),oracle_is_non_executable_GT_diagnostic=True,groups_never_used_by_inference=True,observations_no_GT_queries=sum(r['observations_in_GT']==0 for r in rows),selected_predictions_bitexact=True)
  write(OUT/(ds+'_diagnostic_private.json'),dict(queries=rows,frames=frame_rows));write(PUBLIC/'diagnosis.json',dict(status='partial'if ds=='vidstg'else'complete',datasets=output,old_plan_sha256=h(P/'plan.json'),no_new_GPU_or_parameters=True))
 write(PUBLIC/'diagnosis_seal.json',dict(status='complete',files={'results/tubedetr_boxfixed_followup/diagnosis.json':h(PUBLIC/'diagnosis.json')},posthoc_exposed_diagnosis=True));print(json.dumps(output,indent=2))
if __name__=='__main__':main()
