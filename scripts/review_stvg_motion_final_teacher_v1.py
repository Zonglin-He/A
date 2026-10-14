"""Independent final factorial readback with fixed-layout replacement and full cost."""
import sys,json,re,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_temporal_caption_query_v1 import read,write,sha
B=ROOT/'artifacts/stvg_motion_guided_instance_opd_v1';D=B/'display_order_layout001'
def main():
 rows=[]
 for base,count in [(B,96),(D,32)]:
  bar=read(base/'TEACHER_GLOBAL_BARRIER.json');rt=read(base/'RUNTIME.json');jobs={j['job_id']:j for j in read(base/'TEACHER_JOB_LOCK.json')['records']};assert bar['status']=='complete' and bar['count']==len(jobs)==count and not bar['GT_read'];assert bar['runtime_sha256']==sha(base/'RUNTIME.json') and bar['job_lock_sha256']==sha(base/'TEACHER_JOB_LOCK.json') and bar['input_barrier_sha256']==sha(base/'CANDIDATE_INPUT_BARRIER.json')
  for p,h in {**rt['pins'],**rt['protected']}.items():assert sha(ROOT/p)==h,p
  assert len(bar['files'])==count and len({f['path']for f in bar['files']})==count
  seen=set()
  for f in bar['files']:
   p=ROOT/f['path'];assert sha(p)==f['sha256'] and p.stat().st_size==f['bytes'];z=read(p);assert z['job_id']not in seen;seen.add(z['job_id']);j=jobs[z['job_id']];assert z['input_videos']==j['videos'] and z['candidate_bindings']==j['candidate_bindings'] and z['prompt_sha256']==hashlib.sha256(j['prompt'].encode()).hexdigest()==j['prompt_sha256'] and z['source_case_sha256']==j['source_case_sha256'] and not z['GT_read']
   raw=re.sub(r'^```(?:json)?\s*','',z['raw_output'].strip());raw=re.sub(r'\s*```$','',raw)
   try:
    v=json.loads(raw)['id'];assert type(v)is int and v in j['allowed_output_ids'];valid=True;selected=j['visible_id_to_candidate'].get(str(v))
   except Exception:valid=False;v=None;selected=None
   assert (z['parsed']['status']=='valid')==valid and z['parsed']['selected_candidate_id']==selected
   rows.append(dict(case_id=z['case_id'],dataset=z['dataset'],cohort=j['cohort'],kind=z['kind'],corrected_layout=base==D,format_valid=valid,visible_id=v,selected_candidate_id=selected,truncated=z['output_truncated_at_cap'],input_tokens=z['input_tokens'],visual_tokens=z['visual_tokens'],output_tokens=z['output_tokens'],end_to_end_request_seconds=z['end_to_end_request_seconds'],model_generate_seconds=z['model_forward_generate_seconds'],GPU_peak_bytes=z['GPU_peak_bytes']))
  assert seen==set(jobs)
 effective=[r for r in rows if r['corrected_layout']or r['kind']not in ['order_swap','joint_swap']];assert len(effective)==96
 inp=read(B/'CANDIDATE_INPUT_BARRIER.json')['records'];refs={r['case_id']:r for r in read(B/'METHOD_REFERENCE_LOCK.json')['records']};cases=[]
 for c in inp:
  rr={r['kind']:r for r in effective if r['case_id']==c['case_id']};assert len(rr)==6;s=rr['base']['selected_candidate_id'];assert s==refs[c['case_id']]['references']['teacher']['selected_candidate_id'];checks={k+'_same_geometry':rr['base']['format_valid']and rr[k]['format_valid']and s==rr[k]['selected_candidate_id']for k in ['id_swap','color_swap','order_swap','joint_swap']}
  cases.append(dict(case_id=c['case_id'],dataset=c['dataset'],cohort=c['cohort'],base_selected_candidate_id=s,no_update=refs[c['case_id']]['references']['teacher']['no_update'],reference_count=len(refs[c['case_id']]['references']['teacher']['anchors']),**checks,wrong_query_choice_changed=s!=rr['wrong_query']['selected_candidate_id'],wrong_query_abstain=rr['wrong_query']['format_valid']and rr['wrong_query']['selected_candidate_id']is None,wrong_query_different_nonzero=rr['wrong_query']['format_valid']and rr['wrong_query']['selected_candidate_id']is not None and s!=rr['wrong_query']['selected_candidate_id']))
 def cost(rr):return {**{k:sum(r[k]for r in rr)for k in ['input_tokens','visual_tokens','output_tokens','end_to_end_request_seconds','model_generate_seconds']},'GPU_peak_bytes':max(r['GPU_peak_bytes']for r in rr)}
 out=dict(status='pass',scope='response legality, input/source bindings and factorial readback; not semantic correctness',actual_call_count=128,effective_call_count=96,actual_rows=rows,effective_rows=effective,cases=cases,actual_cost=cost(rows),effective_cost=cost(effective),superseded_layout_control_count=32,format_valid_count=sum(r['format_valid']for r in rows),GT_read=False,method_reference_lock_unchanged=True)
 write(B/'ROOT_TEACHER_FINAL_READBACK003.json',out);print({k:sum(c[k]for c in cases)for k in ['id_swap_same_geometry','color_swap_same_geometry','order_swap_same_geometry','joint_swap_same_geometry','wrong_query_choice_changed','no_update']});print('valid',out['format_valid_count'])
if __name__=='__main__':main()
