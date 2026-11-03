"""Independent all-case index, weight, frozen readout and UNKNOWN audit."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def exact(a,b):assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan)
 for rel,h in {**p['programs_sha256'],**p['inputs_sha256']}.items():assert sha(ROOT/rel)==h
 pred=ROOT/p['prediction_root'];seal=read(pred/'SEALED.json');rows=read(ROOT/p['manifest']);arms=seal['arm_names'];statuses=read(pred/'READOUT_STATUS_PRIVATE.json');assert [r['id'] for r in rows]==[r['id'] for r in statuses]
 eq=read(ROOT/p['equation_audit']);assert eq['state'].startswith('PASS_') and eq['saved_native_replay_bitexact']
 for rel,h in seal['files'].items():assert sha(pred/rel)==h
 latroot=ROOT/p['capture_root'];ls=read(latroot/'SEALED.json');groot=ROOT/p['grouped_root'];gs=read(groot/'SEALED.json');canon=ROOT/p['canonical_root'];cs=read(canon/'SEALED.json');start=time.perf_counter();bad=0;weight_vectors=0;maxerr=0.;all_stats=[]
 for row,st in zip(rows,statuses):
  sid=row['id'];rel='samples_PRIVATE/'+sid+'.npz'
  for folder,ss in [(latroot,ls),(groot,gs),(canon,cs)]:assert sha(folder/rel)==ss['files'][rel]
  with np.load(latroot/rel,allow_pickle=False) as l,np.load(groot/rel,allow_pickle=False) as g,np.load(canon/rel,allow_pickle=False) as c,np.load(pred/rel,allow_pickle=False) as z,np.load(groot/'carriers_PRIVATE'/(sid+'.npz'),allow_pickle=False) as carrier,np.load(pred/'supports_PRIVATE'/(sid+'.npz'),allow_pickle=False) as support:
   exact(z['source_boxes'],c['boxes']);exact(z['source_scores'],c['scores']);exact(l['boxes'],c['boxes']);exact(l['scores'],c['scores']);exact(l['prefix_vis'][0],l['seed_features'].T.copy());exact(l['prefix_lang'][0],l['text_memory']);exact(l['decoder_query_5'],l['h'])
   assert np.array_equal(l['input_sample_xyz'][l['composed_fp2_input_indices']],l['fp2_xyz']);assert np.array_equal(l['point_indices'][l['composed_fp2_input_indices']],l['seed_raw_indices']);assert np.array_equal(l['fp2_xyz'][l['query_points_sample_inds']],l['query_anchor_xyz']);bad+=not np.array_equal(l['input_sample_xyz'][l['native_fp2_input_indices']],l['fp2_xyz'])
   case=read(pred/'cases_PRIVATE'/(sid+'.json'));orig=read(groot/'attributes_PRIVATE'/(sid+'.json'))['text_graph_unchanged'];spans={'T':orig['base_text']['subject']};spans.update({r['entity']:{'span':r['span'],'text':r['phrase']} for r in orig['references']});offsets=np.asarray(read(latroot/'cases_PRIVATE'/(sid+'.json'))['token_offsets_PRIVATE']);masks={}
   for ent,spec in spans.items():
    lo,hi=spec['span'];fragment=row['caption'][lo:hi];literal=spec['text'];occ=[j for j in range(len(fragment)) if fragment.startswith(literal,j)];mask=np.zeros(len(offsets),bool)
    if len(occ)==1:
     left=lo+occ[0];right=left+len(literal);other=row['caption'][lo:left]+row['caption'][right:hi]
     if not any(ch.isalnum() for ch in other):mask=(offsets[:,0]<right)&(offsets[:,1]>left)&(offsets[:,1]>offsets[:,0])
    masks[ent]=mask;assert case['entity_masks'][ent]['available']==bool(mask.any())
   pre=l['fp2_features'].T.astype('float64');pre/=np.maximum(np.sqrt((pre*pre).sum(1))[:,None],1e-12);qp=l['query_point_5'].astype('float64');qt=l['query_token_5'].astype('float64');tp=l['token_point_2'].astype('float64')
   for name in ['query_point_5','query_token_5','token_point_2']:
    m=l[name];assert np.isfinite(m).all() and (m>=0).all() and np.allclose(m.sum(1),1,atol=3e-6,rtol=0)
   seedless=0
   for m,q in enumerate(carrier['group_medoid_queries']):
    q=int(q);nn=support['nearest_seed__'+str(m)];n=int(carrier['instance_crop_mask_pixel_offsets'][m+1]-carrier['instance_crop_mask_pixel_offsets'][m]);box=c['boxes'][q];available=((np.abs(l['fp2_xyz'].astype(float)-box[:3])<=box[3:]/2).all(1) if (box[3:]>0).all() else np.zeros(1024,bool));has=bool(available.any());seedless+=not has
    if has:assert len(nn)==n and available[nn].all()
    else:assert len(nn)==0
    anchor=int(l['query_points_sample_inds'][q]);prew=np.clip((pre@pre[anchor]+1)/2,0,1);perm=np.random.default_rng(int.from_bytes(hashlib.sha256(('R53_seed_mapping_control_v1:'+sid+':'+str(q)).encode()).digest()[:8],'little')).permutation(1024)
    for ent,mask in masks.items():
     sem=None
     if mask.any() and qt[q,mask].sum()>0:
      entity_point=np.sum(tp[mask]*(qt[q,mask]/qt[q,mask].sum())[:,None],axis=0);sem=np.sqrt(np.maximum(qp[q]*entity_point,0));sem=sem/(sem.sum() or 1)*1024
     for variant in ['pre','native','shuffle']:
      if not has or (variant!='pre' and sem is None):expected=np.ones(n)
      else:expected=(prew if variant=='pre' else sem if variant=='native' else sem[perm])[nn]
      actual=support[variant+'__'+ent+'__'+str(m)];assert actual.shape==(n,) and np.isfinite(actual).all() and (actual>=0).all();error=float(np.max(np.abs(actual-expected))) if n else 0.;assert error<1e-10;maxerr=max(maxerr,error);weight_vectors+=1
   assert seedless==case['counts']['no_seed_groups'];assert case['counts']['geometry_descriptor_max_error']<1e-11
   assert z['arm_names'].tolist()==arms
   for j,arm in enumerate(arms):
    if arm=='source':expected=c['scores'].astype(float)
    elif arm=='external_G':expected=z['scores'][j]
    else:
     family,kind=arm.split('_');cc=g['components'][0] if family=='geometry' else z['components'][['pre','native','shuffle'].index(family)];subset='VPR' if kind=='full' else kind;expected=c['scores'].astype(float)
     for idx,cue in enumerate('VPR'):
      if cue in subset:expected=expected+cc[idx]*2
    exact(z['scores'][j],expected);assert z['queries'][j]==int(expected.argmax());exact(z['selected_original_boxes'][j],c['boxes'][int(expected.argmax())]);status=st['arm_statuses'][arm];assert status['no_parameter_update'] and all(type(v) is bool for v in status.values())
    if arm.startswith(('pre_','native_','shuffle_')) and seedless:assert status['fallback']
    if arm.startswith(('native_','shuffle_')) and not case['entity_masks']['T']['available']:assert status['unknown'] and status['fallback']
   all_stats.append({'platform':row['platform'],'groups':len(carrier['group_medoid_queries']),'seedless_groups':seedless,'T_mask_available':case['entity_masks']['T']['available'],'R_masks_available':sum(v['available'] for k,v in case['entity_masks'].items() if k!='T'),'R_masks_total':len(case['entity_masks'])-1,'separator_only_bridges':sum(v['narrowed_separator_only'] for v in case['entity_masks'].values())})
 result={'state':'PASS_R53_INDEPENDENT_INDEX_SUPPORT_AND_FROZEN_READOUT_AUDIT','n':len(rows),'arm_names':arms,'prediction_seal_sha256':sha(pred/'SEALED.json'),'canonical_source_256_boxes_and_readout_verified':True,'all_unknown_fallback_and_zero_update_rows_retained':True,'native_fp2_index_bad_cases':bad,'all_composed_index_coordinates_exact':True,'all_query_anchor_seed_coordinates_exact':True,'latent_weight_vectors_checked':weight_vectors,'maximum_independent_weight_error':maxerr,'seconds':time.perf_counter()-start,'equation_audit_sha256':sha(ROOT/p['equation_audit']),'new_GT_or_model_or_media_reads':0}
 out=ROOT/'research/round53/prediction_audit_attempt001';assert not out.exists();out.mkdir();(out/'AUDIT_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');(out/'COVERAGE_STATS.json').write_text(json.dumps(all_stats,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
