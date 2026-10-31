"""Fixed pre-label role-gate ablations of the same actual-pair calls.
Ranking a loser lower is not declaring it a proven non-target instance.
"""
import sys,time
from common import ROOT,EXP,read,dump,sha,seal,verify,check_plan
POLICIES=('strict_roles','winner_role','pair_only')
def variant(pair_record,roles,scores,policy):
 p=pair_record['pair'];base=pair_record['variants']['main'];out=dict(base);out['policy']=policy;out['roles_original']=dict(base['roles'])
 if not base['target_consensus']:return out
 winner=base['positive_mask'];loser=base['negative_mask']
 if policy=='strict_roles' and (roles[str(winner)]!='T' or roles[str(loser)] not in ('R','O')):return out
 if policy=='winner_role' and roles[str(winner)]!='T':out['reason']='winner_role_unconfirmed';return out
 ids=p['mask_indices'];pos=p['query_groups'][ids.index(winner)];neg=p['query_groups'][ids.index(loser)]
 out['role_accepted']=True
 if not p['geometry_separated'] or not pos or not neg:out['reason']='pair_to_3D_mapping_not_supported';return out
 assert not set(pos)&set(neg)
 q0=int(scores.argmax());qpos=max(pos,key=lambda q:(float(scores[q]),-q));qneg=max(neg,key=lambda q:(float(scores[q]),-q))
 out.update(status='usable',reason='actual_pair_relative_preference_'+policy,mapping_accepted=True,positive_queries=pos,negative_queries=neg,pair_positive_query=qpos,pair_negative_query=qneg,direct_query=qpos if q0 in neg else q0,relation_status='full_caption_pair_comparison_not_global_identity_proof',loser_is_relative_not_verified_negative=True)
 return out
def main():
 import numpy as np
 check_plan();lock=read(EXP/'ROLE_GATE_PRELABEL_ADDENDUM.json')
 for rel,h in lock['programs'].items():assert sha(ROOT/rel)==h
 source=ROOT/'crossplatform/round17/G64_attempt001';verify(source);out=ROOT/'crossplatform/round17/G_variants64_attempt001';out.mkdir(exist_ok=False);(out/'cases_PRIVATE').mkdir();counts={p:0 for p in POLICIES};start=time.perf_counter()
 for row in read(EXP.parent/'input/A64_PRIVATE.json'):
  original=read(source/'cases_PRIVATE'/(row['id']+'.json'))
  with np.load(ROOT/'crossplatform/round17/features96_attempt001/samples_PRIVATE'/(row['id']+'.npz'),allow_pickle=False) as z:scores=z['scores'].copy()
  variants={};allpairs=[]
  for p in original['pairs']:allpairs.append({'pair_alias':p['pair']['pair_alias'],'variants':{policy:variant(p,original['role_answers'],scores,policy) for policy in POLICIES}})
  for policy in POLICIES:
   accepted=[p['variants'][policy] for p in allpairs if p['variants'][policy]['status']=='usable'];variants[policy]=accepted[0] if accepted else {'status':'unusable','reason':'no_accepted_pair_'+policy,'positive_queries':[],'negative_queries':[],'direct_query':int(scores.argmax()),'relation_status':'unknown'};counts[policy]+=bool(accepted)
  dump(out/'cases_PRIVATE'/(row['id']+'.json'),{'id_PRIVATE':row['id'],'alias':original['alias'],'platform':row['platform'],'subset':original['subset'],'variants':variants,'all_pairs':allpairs})
 dump(out/'registered_prelabel_addendum.json',lock);seal(out,'R17_role_gate_variants_sealed_before_current_A_labels',n=64,accepted_observation_counts=counts,original_G_seal_sha256=sha(source/'SEALED.json'),seconds=time.perf_counter()-start,programs=lock['programs']);print('G_VARIANTS_SEALED',counts,flush=True)
if __name__=='__main__':main()
