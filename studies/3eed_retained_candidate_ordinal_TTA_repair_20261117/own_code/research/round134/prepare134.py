from common134 import *
from ordinal134 import *
import collections,time
def main():
 basecheck();reg=rd(R134/'research/round133/INPUT_REGISTRY_PRIVATE.json');kappa=rd(R134/'research/round132/SOURCE_LOG_CALIBRATION.json')['kappa'];out=[];eng=[];total=0;new=0;maxviol=0.
 for e in reg:
  name=e['source']+'_to_'+e['target'];sid=e['row']['id'];D=R134/'crossplatform/round131/formal'/name
  with np.load(D/'features_PRIVATE'/(sid+'.npz')) as f:pA=f['pA'];pB=f['pB'];valid=f['valid_B']
  scopepath=R134/'crossplatform/round132/log_scopes_PRIVATE'/name/(sid+'.npz')
  with np.load(scopepath) as f:ass=f['assignment'];S=f['S'];C=f['C']
  old=R134/'crossplatform/round133/formal'/name/'samples_PRIVATE'/(sid+'.npz')
  with np.load(old) as f:ps=dict(zip(f['arm_names'],f['scores']));boxes=f['source_boxes']
  groups=[np.flatnonzero(ass==j) for j in range(int(ass.max())+1)];win,lose,margin=evidence(pA,pB,ps['zero_F'],groups,C,kappa)
  ww,wl,perm=wrong_pairs(win,lose,groups,C,e['source']+'|'+sid);effective=bool(not (np.array_equal(ww,win) and np.array_equal(wl,lose)))
  assert not len(win) or (valid.any() and C[win].all() and C[lose].all() and (margin>0).all())
  pp={};projection={};start=time.perf_counter()
  for tag,initial,w,l in [('pair_direct_F','direct_F',win,lose),('partial3_pair_direct_F','partial3_F',win,lose),('wrongpair_direct_F','direct_F',ww,wl)]:
   pp[tag],projection[tag]=project(ps[initial],groups,w,l,margin);maxviol=max(maxviol,projection[tag]['max_violation'])
  P=B134/'evidence_PRIVATE'/name;P.mkdir(parents=True,exist_ok=True);path=P/(sid+'.npz')
  np.savez_compressed(path,assignment=ass,S=S,C=C,pA=pA,pB=pB,win=win,lose=lose,margin=margin,wrong_win=ww,wrong_lose=wl,wrong_perm=perm,**pp)
  item=dict(e,pair_active=bool(len(win)),pair_count=len(win),effective_wrong_pair=effective,evidence_path=str(path.relative_to(R134)),evidence_sha256=sha(path),original_prediction_path=str(old.relative_to(R134)),original_prediction_sha256=sha(old),projection=projection,direct_projection_seconds=time.perf_counter()-start)
  out.append(item);total+=len(win);new+=bool(len(win) and not e['active'])
 for name in sorted({e['source']+'_to_'+e['target'] for e in out}):
  rr=[e for e in out if e['source']+'_to_'+e['target']==name and e['pair_active']]
  for cached in [True,False]:
   # Distinct saved-B / same original-B reconstruction paths, selected without GT.
   ss=[e for e in rr if bool(e['active'] and not e['reference_original_R131'])==cached]
   if ss:eng.append({'source':ss[0]['source'],'target':ss[0]['target'],'id_PRIVATE':ss[0]['row']['id']})
 summary={'n':len(out),'pair_active':sum(e['pair_active'] for e in out),'pair_edges':total,'new_pair_tasks_on_old_inactive':new,'original_partial_active':sum(e['active'] for e in out),'identity_wrong_pair_inputs':sum(e['pair_active'] and not e['effective_wrong_pair'] for e in out),'engineering_n':len(eng),'max_direct_constraint_violation':maxviol,'GT_read':False,'FINAL_access':False}
 dump(H134/'INPUT_REGISTRY_PRIVATE.json',out);dump(H134/'ENGINEERING_INPUTS_PRIVATE.json',eng);dump(H134/'PREPARATION_SUMMARY.json',summary)
 dump(B134/'EVIDENCE_SEALED.json',{'n':len(out),'files':{e['evidence_path']:e['evidence_sha256'] for e in out},'task_GT_used':False,'new_source_calibration':False})
 print(summary,flush=True)
if __name__=='__main__':main()
