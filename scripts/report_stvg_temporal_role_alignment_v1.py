"""Independent saved-scalar audit and anonymous P1 export; no model or GT reads."""
from pathlib import Path
import sys,statistics,copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
BASE=ROOT/'artifacts/stvg_temporal_role_alignment_v1/revision002'
OUT=ROOT/'results/stvg_temporal_role_alignment/2026-10-15'
def main():
 b=BASE/'P1';completion=read(b/'CPU_COMPLETION.json');assert completion['status']=='complete'and completion['count']==64
 rows=read(b/'POSTSEAL_SCORE_ROWS.json')['rows'];saved=read(b/'POSTSEAL_SCORE_SUMMARY.json');ranking=read(b/'POSTSEAL_RANKING_DIAGNOSTICS.json');assert len(rows)==192
 ids=sorted({x['case_id']for x in rows});assert len(ids)==64;mapping={cid:'Q'+str(i+1).zfill(3)for i,cid in enumerate(ids)};by={(x['case_id'],x['arm']):x for x in rows};error=0.
 for x in saved['summary']:
  chosen=[r for r in rows if r['arm']==x['arm']and(x['cohort']=='all64'or r['query_type']==('question'if x['cohort']=='question28'else'caption'))]
  values=[r['metrics'][x['metric']]for r in chosen if r['metrics'][x['metric']]is not None];assert len(values)==x['count']
  error=max(error,abs(statistics.mean(values)-x['mean']))
 for x in saved['paired']:
  delta={r['case_id']:100*(r['metrics'][x['metric']]-by[r['case_id'],'T']['metrics'][x['metric']])for r in rows if r['arm']=='P1'and(x['cohort']=='all64'or r['query_type']==('question'if x['cohort']=='question28'else'caption'))and r['metrics'][x['metric']]is not None and by[r['case_id'],'T']['metrics'][x['metric']]is not None}
  assert delta==x['per_source_delta_pp']and len(delta)==x['count'];error=max(error,abs(statistics.mean(delta.values())-x['mean_delta_pp']))
  assert sum(v>1e-10 for v in delta.values())==x['positive']and sum(v< -1e-10 for v in delta.values())==x['negative']
 assert error<1e-12
 for arm,key in [('P0','P0_accuracy'),('P1','new_accuracy')]:
  valid=[r[key]for r in ranking['records']if r[key]is not None];assert abs(statistics.mean(valid)-ranking['P0_macro_accuracy'if arm=='P0'else'new_macro_accuracy'])<1e-12
 paired=copy.deepcopy(saved['paired'])
 for x in paired:x['per_source_delta_pp']={mapping[k]:v for k,v in x['per_source_delta_pp'].items()}
 anonymous_rows=[dict(source=mapping[r['case_id']],query_type='interrogative'if r['query_type']=='question'else'declarative',arm=r['arm'],metrics=r['metrics'],metric_audit=r['metric_audit'])for r in rows]
 anonymous_ranks=[dict({k:v for k,v in r.items()if k!='case_id'},source=mapping[r['case_id']])for r in ranking['records']]
 job=read(b/'TEACHER_JOB_LOCK.json');inputs=read(b/'INPUT_LOCK.json');assert job['count']==88 and inputs['actual_candidate_count']==106 and inputs['unknown_candidate_count']==18
 p2=read(BASE/'CPU_COMPLETION.json');assert p2['P2_executed']is False and completion['p2_eligible']is False
 write(OUT/'ANONYMOUS_ROWS.json',dict(status='complete',count=192,rows=anonymous_rows))
 write(OUT/'ANONYMOUS_SUMMARIES.json',dict(status='complete',summary=saved['summary'],paired=paired))
 write(OUT/'ANONYMOUS_RANKING.json',dict(status='complete',records=anonymous_ranks,definition=ranking['definition'],P0_macro_accuracy=ranking['P0_macro_accuracy'],P1_macro_accuracy=ranking['new_macro_accuracy'],valid_question_count=sum(r['P0_accuracy']is not None for r in anonymous_ranks),unknown_question_count=sum(r['P0_accuracy']is None for r in anonymous_ranks)))
 bindings={n:sha(b/n)for n in ['ROOT_READBACK.json','GLOBAL_PREDICTION_BARRIER.json','TEACHER_GLOBAL_BARRIER.json','STUDENT_GLOBAL_BARRIER.json','POSTSEAL_SCORE_ROWS.json','POSTSEAL_SCORE_SUMMARY.json','POSTSEAL_RANKING_DIAGNOSTICS.json','CPU_COMPLETION.json']}
 write(OUT/'SOURCE_BINDINGS.json',dict(status='pass',source_receipt_hashes=bindings,runtime_sha256=sha(BASE/'RUNTIME.json'),roster_count=64,question_count=28,caption_count=36,actual_candidate_count=106,unknown_candidate_count=18,uninstantiated_candidate_budget=6,alignment_calls=88,inherited_nonempty_truncated_captions=3,new_caption_calls=0,new_full_Student_GPU_forwards=0,CPU_fit_count=22,CPU_root_repeat_count=22,GT_oracle_fits=0,bootstrap_executed=False,P2_executed=False,method_promoted=False,full_started=False))
 write(BASE/'ROOT_INDEPENDENT_SCALAR_AUDIT002.json',dict(status='pass',scope='saved192 scalar rows, all27 summary cells/9 paired contrasts and ranking means; no new GT evaluation',count=192,summary_count=len(saved['summary']),paired_count=len(paired),maximum_error=error,CPU_only=True,new_GT_read=False,source_hashes=bindings))
 write(OUT/'REPORT_RECEIPT.json',dict(status='complete',scope='finite P1 development comparison only; coordinate37 remains running',row_count=192,summary_count=27,paired_count=9,question_count=28,scalar_max_error=error,P2_eligible=False,P2_executed=False,source_audit_sha256=sha(BASE/'ROOT_INDEPENDENT_SCALAR_AUDIT002.json')))
 print('anonymous P1 export and independent scalar audit pass',len(rows),error)
if __name__=='__main__':main()
