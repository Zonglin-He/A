"""Independent per-stage saved IoU lookup, aggregation, scenes and negatives."""
import argparse,sys,itertools,time
from common68 import *
sys.path.insert(0,str(ROOT/'research/round55'));from audit_saved_scalars55 import aggregate,check_tree
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan);pin(p);start=time.perf_counter();s=ROOT/p['score_root'];ss=read(s/'SEALED.json');tasks=read(s/'ALL_TASK_ROWS_PRIVATE.json');labels=read(ROOT/'research/round55/score_dev336_attempt001/ALL_CANDIDATE_IOUS_PRIVATE.json');manifest=read(ROOT/p['manifest']);arms=p['arm_names'];assert len(tasks)==len(labels)==len(manifest)==336
 for rel,h in ss['files'].items():assert sha(s/rel)==h
 for t,l,r in zip(tasks,labels,manifest):
  assert t['id_PRIVATE']==l['id_PRIVATE']==r['id'] and t['scene_PRIVATE']==r['scene'] and set(t['IoU'])==set(t['queries'])==set(t['statuses'])==set(arms)
  for arm in arms:assert t['IoU'][arm]==l['candidate_IoUs_PRIVATE'][t['queries'][arm]]
 summary,scenes=aggregate(tasks,arms,p['reference_arms'],336);e=check_tree(summary,read(s/'TASK_SUMMARY.json'));es=check_tree(scenes,read(s/'ALL_SCENE_SUMMARIES_PRIVATE.json'));negative=[]
 for t in tasks:
  for arm,ref in itertools.product(arms,repeat=2):
   d=t['IoU'][arm]-t['IoU'][ref]
   if arm!=ref and d< -1e-10:negative.append({'row':t['row'],'part':t['part'],'split':t['split'],'platform':t['platform'],'arm':arm,'reference':ref,'delta_pp':float(d*100)})
 assert negative==read(s/'ALL_ORDERED_NEGATIVE_COMPARISONS.json');out=ROOT/p['scalar_audit_root'];assert not out.exists();out.mkdir(parents=True);result={'state':'PASS_R68_'+p['mode'].upper()+'_INDEPENDENT_TASK_SCENE_NEGATIVE_AUDIT','n':336,'arms':len(arms),'task_values':336*len(arms),'max_summary_error_pp':e,'max_scene_error_pp':es,'all_ordered_negatives':len(negative),'scoring_seal_sha256':sha(s/'SEALED.json'),'program_sha256':sha(__file__),'seconds':time.perf_counter()-start};dump(out/'AUDIT_RESULT.json',result);print(result)
if __name__=='__main__':main()
