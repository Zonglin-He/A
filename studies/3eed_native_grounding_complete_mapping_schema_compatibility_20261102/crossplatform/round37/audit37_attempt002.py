"""Postseal CPU replay of complete mapping, safe parsing and fixed direct readout."""
import numpy as np
from common37 import *
from local_engine37 import check_amendment
def main():
 check_amendment();check_plan();prep=BASE/'prepared208_attempt001';out=BASE/'inference208_attempt002';verify(prep);verify(out);counts=[];calls=0;cases=0
 for r in rows():
  c=prepared(r);z=arrays(r);v=(z['support_count']>=3)&(z['support_coverage']>=.15)&(z['projection_iou']>=.1)&(z['source_boxes'][:,3:]>0).all(1)[:,None];u=v.sum(-1)==1;groups={str(m):np.flatnonzero(v[:,m]&u).tolist() for m in range(len(c['candidates']))};assert groups==c['full_groups'];old=read(ROOT/'crossplatform/round29/groups208_attempt001/cases_PRIVATE'/(r['id']+'.json'))['groups_PRIVATE'];assert old==c['old_subset_groups'];assert all(groups[m]==g for m,g in old.items());source=feature(r);assert np.array_equal(source['boxes'],z['source_boxes']) and np.array_equal(source['scores'],z['source_scores']);rc=read(out/'predictions_PRIVATE'/(r['id']+'.json'));q={'source':int(z['source_scores'].argmax())};legacy=load_npz(ROOT/'crossplatform/round29/direct208_attempt001/predictions_PRIVATE'/(r['id']+'.npz'));names=legacy['arm_names'].tolist();q['R29_direct']=int(legacy['queries'][names.index('full_pair_group')]);q['legacy_G']=int(legacy['queries'][names.index('legacy_G')])
  for interface in ['point','som']:
   f=out/rc['call_refs'][interface]['file'];assert sha(f)==rc['call_refs'][interface]['sha256'];call=read(f);res=call['result'];assert parse_json(res['raw_answer_PRIVATE'],interface)==res['parsed'];assert len(res['image_grid_thw'])==(1 if interface=='point' else 2);assert all(g[0]==1 and g[1]*g[2]//4<=1024 for g in res['image_grid_thw']);assert res['generated_tokens']<=96;assert res['model_resized_wh']==[res['image_grid_thw'][0][2]*14,res['image_grid_thw'][0][1]*14]
   for mapping in ['fullmap','oldsubset']:
    a=direct_readout(interface,res,c,z,subset=mapping=='oldsubset');assert a==rc['readouts'][interface+'_'+mapping];q[interface+'_'+mapping]=a['query']
   calls+=1
  p=load_npz(out/'predictions_PRIVATE'/(r['id']+'.npz'));expected=np.array([q[n] for n in ARMS]);assert p['arm_names'].tolist()==ARMS and np.array_equal(p['queries'],expected) and np.array_equal(p['boxes'],z['source_boxes'][expected]);counts.append({'alias':r['alias'],'source_unchanged':True,'full_old_mapping_and_all_readouts_replayed':True});cases+=1
 ledger=(EXP/'GENERATION_BUDGET.jsonl').read_text().splitlines();assert calls==416 and len(ledger)<=440;dump(EXP/'CPU_AUDIT.json',{'state':'PASSED','cases':cases,'calls':calls,'all_full_mapping_and_oldsubset_exact':True,'all_safe_parse_and_coordinate_metadata_replayed':True,'all7query_arms_and_boxes_exact':True,'generation_attempts':len(ledger),'no_GPU_model_GT_or_parameter_updates':True,'prediction_seal_sha256':sha(out/'SEALED.json'),'plan_sha256':sha(EXP/'PLAN_LOCK.json')});print('AUDIT37_PASSED',cases,calls)
if __name__=='__main__':main()
