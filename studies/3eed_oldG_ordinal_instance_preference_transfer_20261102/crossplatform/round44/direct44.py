"""Phase1 only: fixed oldG, distinct unique groups, no model/GT/new evidence."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[2];EXP=ROOT/'research/round44/phase1';OUT=ROOT/'crossplatform/round44/direct208_attempt001'
ARMS=['source','unrestricted_oldG','eligible_fixed_source_rep']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def get(p,keys=None):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in (keys if keys is not None else z.files)}
def bound_file(folder,rel,seal):
 p=folder/rel;assert sha(p)==seal['files'][rel],str(p);return p

def decide(s,F,groups):
 assert s.shape==F.shape==(256,) and np.isfinite(s).all() and np.isfinite(F).all();q0=int(s.argmax());qG=int(F.argmax());membership={q:[] for q in range(256)}
 for m,g in groups.items():
  assert len(g)==len(set(g)) and all(isinstance(q,int) and 0<=q<256 for q in g)
  for q in g:membership[q].append(str(m))
 m0=membership[q0];mG=membership[qG];G0=groups[m0[0]] if len(m0)==1 else [];GG=groups[mG[0]] if len(mG)==1 else [];rep0=max(G0,key=lambda i:(float(s[i]),-i)) if G0 else None;repG=max(GG,key=lambda i:(float(s[i]),-i)) if GG else None
 if qG==q0:reason='no_corrective_nomination_same_q'
 elif not F[qG]>F[q0]:reason='non_strict_G_tie'
 elif len(m0)!=1:reason='source_query_not_exactly_one_full_group'
 elif len(mG)!=1:reason='G_query_not_exactly_one_full_group'
 elif m0[0]==mG[0]:reason='same_instance_group_no_negative'
 elif set(G0)&set(GG):reason='overlapping_group_members_no_negative'
 else:reason='eligible_distinct_unique_groups'
 eligible=reason=='eligible_distinct_unique_groups';q=int(repG) if eligible else q0
 if G0:assert rep0==q0
 return {'source_q0':q0,'raw_G_q':qG,'G_F64_gap_vs_q0':float(F[qG]-F[q0]),'source_membership_masks':m0,'G_membership_masks':mG,'source_group_members':G0,'G_group_members':GG,'source_group_rep':rep0,'G_group_rep':repG,'eligible':eligible,'reason':reason,'query':q,'accepted_preference':{'positive_queries':GG,'negative_queries':G0,'positive_mask':mG[0],'negative_mask':m0[0],'positive_source_rep':repG,'negative_source_rep':rep0} if eligible else None,'distinct_mask_hypothesis_not_gold_identity':True,'unknown_not_labelled_negative':not eligible}

def main():
 auth=read(EXP/'ROOT_EXECUTION_AUTHORIZATION.json');assert auth['state']=='ROOT_AUTHORIZED_PHASE1_DIRECT_USE_ONLY'
 for f,h in auth['refs_sha256'].items():assert sha(ROOT/f)==h,f
 plan=read(EXP/'PLAN_LOCK.json');assert plan['root_authorization_sha256']==sha(EXP/'ROOT_EXECUTION_AUTHORIZATION.json')
 for f,h in dict(plan['program_sha256'],**plan['input_sha256']).items():assert sha(ROOT/f)==h,f
 gate=read(ROOT/'research/round44/qualification/RESULT.json');assert gate['state']=='PASS_EXACT_SOURCE_ARRAY_COMPATIBILITY_ONLY' and gate['n']==gate['boxes_exact']==gate['scores_exact']==208
 folders={'source':ROOT/'crossplatform/round24/capture336_attempt001','bank':ROOT/'crossplatform/round23/bank272_attempt001','geometry':ROOT/'crossplatform/round37/prepared208_attempt001','history':ROOT/'crossplatform/round29/direct208_attempt001'};seals={k:read(p/'SEALED.json') for k,p in folders.items()};rr=read(ROOT/'research/round24/input/D208_PRIVATE.json');assert len(rr)==208 and len({r['id'] for r in rr})==208;OUT.mkdir(exist_ok=False)
 for n in ['predictions_PRIVATE','cases_PRIVATE']:(OUT/n).mkdir()
 records=[];replays=[]
 for r in rr:
  rel='samples_PRIVATE/'+r['id']+'.npz';src=bound_file(folders['source'],rel,seals['source']);bank=bound_file(folders['bank'],rel,seals['bank']);geom=bound_file(folders['geometry'],'cases_PRIVATE/'+r['id']+'.json',seals['geometry']);hist=bound_file(folders['history'],'predictions_PRIVATE/'+r['id']+'.npz',seals['history']);z=get(src,['scores','boxes']);b=get(bank,['vision','position','relation']);case=read(geom);s=z['scores'];FG=s.astype(float)+2*b['vision']+2*b['position']+2*b['relation'];d=decide(s,FG,case['full_groups']);assert case['source_q0']==d['source_q0']
  for m,g in case['full_groups'].items():
   if g:assert int(case['representatives_full'][m])==max(g,key=lambda i:(float(s[i]),-i))
  hp=get(hist);hi=hp['arm_names'].tolist().index('legacy_G');hq=int(hp['queries'][hi]);hb=hp['boxes'][hi];history_exact=d['raw_G_q']==hq and np.array_equal(z['boxes'][d['raw_G_q']],hb);q=np.array([d['source_q0'],d['raw_G_q'],d['query']],np.int64);np.savez_compressed(OUT/'predictions_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(ARMS),queries=q,boxes=z['boxes'][q],source_scores=s,oldG_F64=FG,historical_oldG_query=np.array(hq),historical_oldG_box=hb)
  replay={'alias':r['alias'],'platform':r['platform'],'unrestricted_F64_query':d['raw_G_q'],'historical_legacy_G_query':hq,'query_exact':d['raw_G_q']==hq,'box_exact':bool(np.array_equal(z['boxes'][d['raw_G_q']],hb)),'combined_exact':history_exact,'F32_teacher_argmax_diagnostic':int(FG.astype(np.float32).argmax()),'F64_F32_argmax_same':int(FG.astype(np.float32).argmax())==d['raw_G_q'],'historical_file':str(hist.relative_to(ROOT)),'historical_sha256':sha(hist)};replays.append(replay);rec={'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'decision':d,'historical_G_replay':replay,'bindings':{'source':{'path':str(src.relative_to(ROOT)),'sha256':sha(src)},'bank':{'path':str(bank.relative_to(ROOT)),'sha256':sha(bank)},'geometry_case':{'path':str(geom.relative_to(ROOT)),'sha256':sha(geom)}},'task_GT_used':False};dump(OUT/'cases_PRIVATE'/(r['id']+'.json'),rec);records.append(rec)
 dump(OUT/'ALL_CASES_PRIVATE.json',records);dump(OUT/'HISTORICAL_G_REPLAY.json',{'state':'EXACT_ALL208' if all(r['combined_exact'] for r in replays) else 'MEASUREMENT_BRIDGE_WARNING_DO_NOT_SILENTLY_REPLACE_REFERENCE','rows':replays,'all208_query_box_exact':all(r['combined_exact'] for r in replays),'F64_F32_argmax_difference_count':sum(not r['F64_F32_argmax_same'] for r in replays),'noGT_selective_readout':True});files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()};dump(OUT/'SEALED.json',{'state':'R44_PHASE1_ALL208_THREEARM_PREDICTIONS_BEFORE_TASKSCORE','files':files,'n':208,'arms':ARMS,'eligible_n':sum(r['decision']['eligible'] for r in records),'oldG_history_exact':all(r['combined_exact'] for r in replays),'root_authorization_sha256':sha(EXP/'ROOT_EXECUTION_AUTHORIZATION.json'),'plan_sha256':sha(EXP/'PLAN_LOCK.json'),'new_G_GPU_VLM_TTA_calls':0,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});print('R44_PHASE1_SEALED','eligible',sum(r['decision']['eligible'] for r in records),'history_exact',all(r['combined_exact'] for r in replays),'F32differences',sum(not r['F64_F32_argmax_same'] for r in replays))
if __name__=='__main__':main()
