"""Literal official float32 IoU vs independent3Dfloat64, separate diagnostic outputs.
Never changes official source. Exceptions swallowed by official helper are counted
through a transparent box3d_iou wrapper that calls/rethrows the original function.
No model/GPU calls; real GT access requires explicitpostseal gate.
"""
from pathlib import Path
import sys,json,hashlib,argparse,importlib.util,warnings,time
import numpy as np
from independent_core_v2 import candidate_ious,transform_gt,validate_native_top10,validate_official_clamp
R=Path(__file__).resolve().parent;B=R.parents[2];U=B/'crossplatform/upstream';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
def load_literal():
 sys.path[:0]=[str(U),str(U/'utils')]
 path=U/'utils/eval_det.py';spec=importlib.util.spec_from_file_location('round5_literal_official_eval_det',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def literal_values(module,boxes,target):
 import torch
 before=module.box3d_iou;events=[];counter=[0]
 def monitored(a,b):
  call=counter[0];counter[0]+=1
  try:return before(a,b)
  except Exception as e:events.append({'candidate_call_offset':call,'exception_type':type(e).__name__,'exception_summary':str(e).splitlines()[0][:150]});raise
 module.box3d_iou=monitored
 try:
  with warnings.catch_warnings(record=True) as caught:
   warnings.simplefilter('always');v,_=module.iou3d_rotated_vs_aligned(torch.as_tensor(np.asarray(target,dtype=np.float32)[None]),torch.as_tensor(np.asarray(boxes,dtype=np.float32)));result=v.detach().cpu().numpy()[0].copy()
 finally:module.box3d_iou=before
 return result,{'n_calls':counter[0],'swallowed_exceptions':events,'warnings':[{'category':x.category.__name__,'summary':str(x.message)[:150]} for x in caught],'nonfinite_indices':np.flatnonzero(~np.isfinite(result)).tolist()}

def validate_cache(z):
 assert z['head_names'].tolist()==['proposal_','0head_','1head_','2head_','3head_','4head_','last_'];assert z['class_names'].tolist()==['car','pedestrian','bus','othervehicle','truck','cyclist'];assert z['boxes'].shape==z['boxes_raw'].shape==(7,256,6);clamp=validate_official_clamp(z['boxes_raw'],z['boxes']);assert z['bbf_scores_by_class'].shape==z['bbs_scores_by_class'].shape==(7,6,256);assert z['class_synonym_found'].shape==z['class_positive_map_nonzero'].shape==(6,);return clamp

def main(args):
 if not args.authorized_postseal:raise RuntimeError('Parent explicitGTauthorization+sealedsource required')
 out=args.output;assert not out.exists();source=args.predictions;sealpath=source/'sealed.json';seal=read(sealpath);assert seal['state']=='sealed_frozen_baseline_no_labels_selected';assert seal['no_actual_target_class_or_numeric_GT_read'] and seal['all_six_candidate_class_scores_precomputed_without_GT_class'];assert sha(args.manifest)==seal['manifest_sha256']
 for n,h in seal['files'].items():assert sha(source/n)==h
 for name,key in [('repeat_control.json','repeat_control_sha256'),('point_provenance.json','point_provenance_sha256'),('trace.json','trace_sha256')]:assert sha(source/name)==seal[key]
 assert sha(args.scoring_index)==args.scoring_index_sha;manifest=read(args.manifest);assert len(manifest)==seal['n']==48,'Literal engineering bridge currentlylockedtosmoke48 only';assert len(seal['files'])==len(manifest)*2
 # Verify every target metadata container by hash without parsing its contents, before first class/box read.
 index={x['id']:x for x in read(args.scoring_index)};wanted={s['id'] for s in manifest};assert wanted<=set(index)
 for sid in wanted:
  row=index[sid];assert sha(row['meta_path'])==row['metadata_container_sha256']
 out.mkdir();module=load_literal();lock={'scorer_sha256':sha(__file__),'independent_core_sha256':sha(R/'independent_core_v2.py'),'independent3D_sha256':sha(B/'tools/obb_iou.py'),'literal_eval_det_sha256':sha(U/'utils/eval_det.py'),'literal_box_util_sha256':sha(U/'utils/box_util.py'),'prediction_seal_sha256':sha(sealpath),'all96cachehashes_and_metadatahashes_before_targetvalueaccess':True,'parent_grant':args.parent_grant,'GPU_model_calls':0,'scope':'smoke48postsealGTbridge; preserve literaldiscrepancies, no candidate/modelchange'};(out/'SCORER_LOCK.json').write_text(json.dumps(lock,indent=2)+'\n');rows=[];start=time.perf_counter()
 for sample in manifest:
  sid=sample['id'];s=index[sid];meta=read(s['meta_path']);obj=meta['ground_info'][s['object_index']];cls=str(obj['class']).lower();target=transform_gt(obj['bbox_3d'],s['platform'],s['waymo_view_suffix']);assert cls in seal['class_names'];ci=seal['class_names'].index(cls)
  for condition in ['full','noun']:
   with np.load(source/(sid+'_'+condition+'.npz'),allow_pickle=False) as cache:z={k:cache[k].copy() for k in cache.files}
   clamp=validate_cache(z);classes=z['class_names'].tolist();assert classes[ci]==cls
   for hi,head in enumerate(z['head_names'].tolist()):
    streams=[('firstnoun_bbf',z['firstnoun_bbf'][hi],z['firstnoun_bbf_native_top10'][hi]),('firstnoun_bbs',z['firstnoun_bbs'][hi],z['firstnoun_bbs_native_top10'][hi])]
    if z['class_synonym_found'][ci]:streams.extend([('official_bbf',z['bbf_scores_by_class'][hi,ci],z['bbf_native_top10_by_class'][hi,ci]),('official_bbs',z['bbs_scores_by_class'][hi,ci],z['bbs_native_top10_by_class'][hi,ci])])
    union=np.unique(np.concatenate([validate_native_top10(sc,ix) for _,sc,ix in streams]));boxes=z['boxes'][hi,union];literal,events=literal_values(module,boxes,target);independent,geom=candidate_ious(boxes,target);lookup={int(q):j for j,q in enumerate(union)}
    for stream,sc,top in streams:
     lv=np.array([literal[lookup[int(q)]] for q in top]);iv=np.array([independent[lookup[int(q)]] for q in top]);threshold_crossings={str(t):np.flatnonzero((lv>t)!=(iv>t)).tolist() for t in [.25,.5]};finite=np.isfinite(lv);rows.append({'id':sid,'platform':s['platform'],'scene':s['scene'],'condition':condition,'head':head,'stream':stream,'n':10,'literal_top1_iou':float(lv[0]) if finite[0] else None,'audited_top1_iou':float(iv[0]),'literal_best_top5_iou':float(np.max(lv[:5])) if np.isfinite(lv[:5]).all() else None,'audited_best_top5_iou':float(iv[:5].max()),'literal_best_top10_iou':float(np.max(lv)) if finite.all() else None,'audited_best_top10_iou':float(iv.max()),'max_abs_finite_iou_difference':float(np.max(abs(lv[finite]-iv[finite]))) if finite.any() else None,'strict_threshold_crossing_native_top10_positions':threshold_crossings,'literal_nonfinite_count':int((~finite).sum()),'GTclass_map_nonzero':bool(z['class_positive_map_nonzero'][ci]),'literal_pair_engineering':events,'clamp_receipt':clamp})
  print('LITERAL_BRIDGED',sid,flush=True)
 finite=[r['max_abs_finite_iou_difference'] for r in rows if r['max_abs_finite_iou_difference'] is not None];result={'scope':'literalofficialfloat32 vsindependentfloat64; never substitute oneforother','n_parents':len(manifest),'n_stream_head_condition_rows':len(rows),'max_abs_finite_difference':max(finite),'n_rows_with_strict_threshold_disagreement':sum(any(r['strict_threshold_crossing_native_top10_positions'][str(t)] for t in [.25,.5]) for r in rows),'n_rows_with_nonfinite_literal':sum(r['literal_nonfinite_count']>0 for r in rows),'rows':rows,'seconds':time.perf_counter()-start,'warnings':['Exception events repeated perstreamwhen sharedqueryunion; deduplicate byid/condition/head/calloffset beforetotals','Literalnonfinite preserved asnull metadata; notnanmean/zero silently','This bridge is anengineeringdiagnostic afterpredictionseal, not an independentheldoutmethodtest']};(out/'RESULTS_PRIVATE.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');(out/'SEALED.json').write_text(json.dumps({'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}},indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--predictions',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--scoring-index',type=Path,required=True);ap.add_argument('--scoring-index-sha',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--authorized-postseal',action='store_true');ap.add_argument('--parent-grant',required=True);main(ap.parse_args())
