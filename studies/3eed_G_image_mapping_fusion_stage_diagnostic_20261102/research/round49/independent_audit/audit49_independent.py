"""Post-seal independent old336 R49 source/W/stage/16-readout audit. No GT/media."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import json,hashlib,time,datetime
from pathlib import Path
import numpy as np
from reference49 import ARMS,extract_original_relation,reference_stage,reference_readouts
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;EXP=ROOT/'research/round49/experiment';P=ROOT/'crossplatform/round49/stage336_attempt001';CAP=ROOT/'crossplatform/round24/capture336_attempt001'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def exact(a,b,label):
 a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b) and a.tobytes()==b.tobytes(),label

def verify(folder,sealname='SEALED.json'):
 s=read(folder/sealname)
 for n,h in s['files'].items():assert not Path(n).is_absolute() and '..' not in Path(n).parts and sha(folder/n)==h,(folder,n)
 return s

def code_check():
 l=read(HERE/'INDEPENDENT_CODE_LOCK.json')
 for n,h in {**l['program_sha256'],**l['reference_input_sha256']}.items():assert sha(ROOT/n)==h,n

def old_rows():
 rr=read(ROOT/'research/round24/input/ALL336_PRIVATE.json');split={}
 for s in ['D208','V64','M64']:
  for r in read(ROOT/'research/round24/input'/(s+'_PRIVATE.json')):assert r['id'] not in split;split[r['id']]=s
 assert len(rr)==len(split)==336 and {r['id'] for r in rr}==set(split);return rr,split

def original_source(r):
 directory={'A':'round17/features96_attempt001','H':'round19/sourceH32_attempt001','V':'round19/sourceV16_attempt001','W':'round20/sourceW32_attempt001','Z':'round20/sourceZ64_attempt001','K':'round23/sourceK64_attempt001','M':'round24/capture336_attempt001'}[r['stage']]
 return ROOT/'crossplatform'/directory/'samples_PRIVATE'/(r['id']+'.npz')

def main():
 start=time.perf_counter();code_check();auth=read(EXP/'ROOT_INDEPENDENT_AUDIT_AUTHORIZATION.json');assert auth['state']=='ROOT_AUTHORIZED_R49_INDEPENDENT_CACHED_CPU_AUDIT_NO_GT';assert auth['audit_program_sha256']==sha(__file__) and auth['independent_code_lock_sha256']==sha(HERE/'INDEPENDENT_CODE_LOCK.json');plan=read(EXP/'STAGE_PLAN_LOCK.json');assert auth['stage_plan_sha256']==sha(EXP/'STAGE_PLAN_LOCK.json') and auth['prediction_seal_sha256']==sha(P/'SEALED.json');seal=verify(P);assert seal['plan_sha256']==auth['stage_plan_sha256'] and seal['n']==336 and seal['arms']==ARMS
 for n,h in {**plan['programs_sha256'],**plan['inputs_sha256']}.items():assert sha(ROOT/n)==h,n
 inventory=ROOT/'research/round49/inventory_attempt001';verify(inventory);bindings=read(inventory/'SOURCE_BINDINGS_PRIVATE.json');saved=read(P/'INPUT_BINDINGS_PRIVATE.json');assert all(saved[k]==v for k,v in bindings.items() if k in saved)
 for n,h in {**bindings,**saved}.items():assert not Path(n).is_absolute() and '..' not in Path(n).parts and sha(ROOT/n)==h,n
 relation=extract_original_relation((ROOT/'crossplatform/round23/rank23.py').read_text(),(ROOT/'crossplatform/round20/detect.py').read_text());rr,split=old_rows();auditrows=[];sourcechecks=[];rawgraymax=0.;clipall={k:0 for k in ['vision','position','relation']}
 for r in rr:
  sid=r['id'];suffix='M64' if r['stage']=='M' else '272';base=ROOT/'crossplatform'/('round24' if r['stage']=='M' else 'round23');z=load(CAP/'samples_PRIVATE'/(sid+'.npz'));arr=load(base/('prepared'+suffix+'_attempt001')/'arrays_PRIVATE'/(sid+'.npz'));prep=read(base/('prepared'+suffix+'_attempt001')/'cases_PRIVATE'/(sid+'.json'));ev=read(base/('evidence'+suffix+'_attempt001')/'cases_PRIVATE'/(sid+'.json'));text=read(base/('text'+suffix+'_attempt001')/'cases_PRIVATE'/(sid+'.json'));bank=load(base/('bank'+suffix+'_attempt001')/'samples_PRIVATE'/(sid+'.npz'));pred=load(P/'samples_PRIVATE'/(sid+'.npz'));case=read(P/'cases_PRIVATE'/(sid+'.json'));pool=prep['mask_pool_PRIVATE'];W=arr['weights']
  assert case['id_PRIVATE']==sid and case['platform']==r['platform'] and case['split']==split[sid] and case['scene_PRIVATE']==r['scene'];assert W.shape==(256,len(pool)) and np.isfinite(W).all() and (W>=0).all() and len(pool)==len(set(pool));exact(z['boxes'],arr['source_boxes'],'prepared source boxes');exact(z['scores'],arr['source_scores'],'prepared source scores')
  orig=load(original_source(r));r25=load(ROOT/'crossplatform/round25/source_features_attempt002/samples_PRIVATE'/(sid+'.npz'));oldjoint=load(ROOT/'crossplatform/round27/scores528_attempt001/samples_PRIVATE'/(sid+'.npz'))
  for label,a in [('original',orig),('R25',r25),('joint',{'boxes':oldjoint['boxes'],'scores':oldjoint['source']})]:
   exact(z['boxes'],a['boxes'],label+' boxes');exact(z['scores'],a['scores'],label+' scores')
  sourcechecks.append({'alias':r['alias'],'platform':r['platform'],'split':split[sid],'R24_original_prepared_R25_joint_boxes_scores_bitexact':True})
  mp=ROOT/case['mask_cache_path_PRIVATE'];assert sha(mp)==case['mask_cache_sha256']==prep['bindings']['mask_cache_sha256'];assert mp.parent in [ROOT/'crossplatform/round15/masks64_attempt001',ROOT/'crossplatform/round12/masks501_attempt001'];ms=read(mp.parent/'SEALED.json');assert ms['files'][mp.name]==sha(mp);masks=load(mp);stage=reference_stage(z['scores'],W,pool,ev,text,masks,relation)
  assert set(ev['targets_PRIVATE'])=={str(m) for m in pool}
  for m in pool:
   views=ev['targets_PRIVATE'][str(m)];assert len(views)==2
   for v in views:
    delta=v['full']['log_odds_yes_no']-v['gray']['log_odds_yes_no'];rawgraymax=max(rawgraymax,abs(delta-v['visual_evidence']));assert delta==v['visual_evidence']
  for k in ['vision','position','relation']:exact(stage[k],bank[k],k+' original bank');clipall[k]+=int((stage[k]!=stage[k+'_unclipped']).sum())
  for k in ['mapped','image_e','image_full','image_gray','vv','rr','mask_relations','vision','position','relation','vision_unclipped','position_unclipped','relation_unclipped']:exact(stage[k],pred[k],k+' stage')
  assert stage['anchor_v']==case['anchor_v'];assert (stage['anchor_r'] is None and case['anchor_r'] is None) or np.array_equal(stage['anchor_r'],case['anchor_r']);assert case['subject_query_PRIVATE']==text['subject_query'];assert case['visual_text_source']['string_sha256']==hashlib.sha256(text['subject_query'].encode()).hexdigest();assert case['joint_text_source']['string_sha256']==hashlib.sha256(r['caption'].encode()).hexdigest()
  joint=read(ROOT/'crossplatform/round27/joint528_attempt001/cases_PRIVATE'/(sid+'.json'));assert set(joint['targets_PRIVATE'])=={str(m) for m in pool};ej=np.array([joint['targets_PRIVATE'][str(m)]['crop_scene_joint']['log_odds_yes_no'] for m in pool]);scores,domains,q,ties,fb,jcue,jv,fv=reference_readouts(z['scores'],stage,ej);assert pred['arm_names'].tolist()==ARMS;exact(scores,pred['arm_scores'],'all16 full score vectors');exact(domains,pred['query_domains'],'16 domains');exact(q,pred['queries'],'16 winners');exact(z['boxes'][q],pred['selected_boxes'],'16 unchanged source boxes');exact(z['boxes'],pred['source_boxes'],'source pool');exact(z['scores'],pred['source_scores'],'source score');assert ties==case['max_tie_sets_PRIVATE'] and fb==case['empty_mapped_source_fallback']
  for expected,k in [(ej,'joint_e'),(jcue,'joint_rawcue'),(jv,'joint_operator_vision'),(fv,'full_yes_operator_vision'),(W,'weights')]:exact(expected,pred[k],k)
  j=oldjoint['cue_names'].tolist().index('crop_scene_joint');exact(jcue,oldjoint['cues'][j],'historical joint rawcue');oldfolder=ROOT/'crossplatform/round24'/('finalM64_attempt001' if split[sid]=='M64' else 'modal_'+split[sid]+'_attempt001');oldseal=read(oldfolder/'SEALED.json');op=oldfolder/'predictions_PRIVATE'/(sid+'.npz');assert oldseal['files'][op.relative_to(oldfolder).as_posix()]==sha(op);old=load(op);ix=old['arm_names'].tolist().index('Qwen_G_direct');assert int(old['query_indices'][ix])==int(q[7]);exact(old['boxes'][ix],pred['selected_boxes'][7],'historical G selected box')
  auditrows.append({'alias':r['alias'],'id_PRIVATE':sid,'split':split[sid],'platform':r['platform'],'scene_PRIVATE':r['scene'],'pool_instances':len(pool),'mapped_query_count':int(stage['mapped'].sum()),'q0_mapped':bool(stage['mapped'][int(z['scores'].argmax())]),'no_mapped_queries':not bool(stage['mapped'].any()),'max_tie_sizes':{a:len(ties[j]) for j,a in enumerate(ARMS)},'empty_domain_fallback':dict(zip(ARMS,fb)),'clip_counts':{k:case[k+'_clip_n'] for k in ['vision','position','relation']},'all_16_scalar_arithmetic_and_boxes_bitexact':True})
 dump(HERE/'ALL336_STAGE_AUDIT_FLAGS_PRIVATE.json',auditrows);dump(HERE/'ALL336_SOURCE_BRIDGES_PRIVATE.json',sourcechecks);result={'state':'PASS_R49_INDEPENDENT_CACHED_STAGE_AND_READOUT_AUDIT_NO_GT','n':336,'arms':16,'full_score_values_bitexact':336*16*256,'selected_query_box_values_bitexact':336*16,'all_five_source_bridges_bitexact':True,'original_V_P_R_bank_all336_bitexact':True,'legacy_historical_all336_qbox_exact':True,'joint_historical_all336_cue_exact':True,'full_gray_subtraction_maxerror':rawgraymax,'clip_total_query_coordinates':clipall,'source_scene_mask_evidence_reference_sha_verified':True,'own_new_readouts_compared_with_frozen_original_pure_relation_reference':True,'GT_bbox_RGB_rawXYZ_model_optimizer_or_FINAL_read':False,'prediction_seal_sha256':sha(P/'SEALED.json'),'stage_plan_sha256':sha(EXP/'STAGE_PLAN_LOCK.json'),'program_sha256':sha(__file__),'seconds':time.perf_counter()-start};dump(HERE/'AUDIT_RESULT.json',result);print(json.dumps(result))
if __name__=='__main__':main()
