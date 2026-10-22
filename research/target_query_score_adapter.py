"""CPU-only postseal qualification-name repair, preserving immutable inference code."""
import os,sys,time,ast,hashlib,json
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from research.native_event_target_query_v1.run import D,P,read,put,sha,verify

def main():
 p=verify();s=read(D/'sealed.json');assert s['count']==128 and not s['GT_read']and s['plan_sha256']==sha(D/'plan.json');assert not(D/'score_receipt.json').exists()
 for e in s['files'].values():assert sha(D/e['path'])==e['sha256']and sha(D/e['trace_path'])==e['trace_sha256']
 f=R/'research/native_event_target_query_v1/score.py';old=f.read_text();before="['RandomTargetQuery','WrongTimeSemanticLoop2','SharpenOriginal','ReflectedSemanticLoop2','ParentGlobalSemanticLoop2']";after="['ReflectedQuery','GlobalSemanticQuery']";assert old.count(before)==1;new=old.replace(before,after);assert 'beyondReflectedQuery/GlobalSemanticQuery ANDsameinformationDirectTargett/fullv'in p['qualification']
 # Only one literal list of stale comparison names changes; metric/action/sample code identical.
 a=ast.parse(old);b=ast.parse(new);aa=[n for n in ast.walk(a)if isinstance(n,ast.List)and ast.get_source_segment(old,n)==before];bb=[n for n in ast.walk(b)if isinstance(n,ast.List)and ast.get_source_segment(new,n)==after];assert len(aa)==len(bb)==1
 aa[0].elts=[];bb[0].elts=[];assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
 put(D/'score_adapter_preflight.json',dict(adapter_sha256=sha(Path(__file__)),original_frozen_score_sha256=sha(f),seal_sha256=sha(D/'sealed.json'),before=before,after=after,only_qualification_name_list_AST_changed=True,predeclared_plan_criteria_unchanged=True,all128_prediction_and_trace_hashes_verified=True,new_model_calls=0,new_predictions=0));(D/'scoring_adapter_snapshot.py').write_text(new)
 scope=dict(__name__='__main__',__file__=str(f));exec(compile(new,str(f)+'[postseal-name-adapter]','exec'),scope)
 assert all(sha(D/e['path'])==e['sha256']and sha(D/e['trace_path'])==e['trace_sha256']for e in s['files'].values());put(D/'score_adapter_receipt.json',dict(adapter_sha256=sha(Path(__file__)),seal_sha256=sha(D/'sealed.json'),original_frozen_score_sha256=sha(f),report_sha256=sha(P/'report.json'),all128_predictions_preserved=True,inference_plan_unchanged=True,GT_only_postseal_CPU=True))
if __name__=='__main__':main()
