"""Read every sealed response independently, lock references before labels."""
import sys,json,hashlib,re,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_temporal_caption_query_v1 import read,write,sha
B=ROOT/'artifacts/stvg_motion_guided_instance_opd_v1'
def main():
 bar=read(B/'TEACHER_GLOBAL_BARRIER.json');rt=read(B/'RUNTIME.json');jobs=read(B/'TEACHER_JOB_LOCK.json')['records'];assert bar['count']==len(jobs)==96 and bar['GT_read'] is False
 assert bar['runtime_sha256']==sha(B/'RUNTIME.json') and bar['input_barrier_sha256']==sha(B/'CANDIDATE_INPUT_BARRIER.json') and bar['job_lock_sha256']==sha(B/'TEACHER_JOB_LOCK.json')
 for p,h in {**rt['pins'],**rt['protected']}.items():assert sha(ROOT/p)==h,p
 index={j['job_id']:j for j in jobs};rows=[];cost=dict(input_tokens=0,visual_tokens=0,output_tokens=0,end_to_end_request_seconds=0.,model_generate_seconds=0.,GPU_peak_bytes=0)
 for f in bar['files']:
  p=ROOT/f['path'];assert sha(p)==f['sha256'] and p.stat().st_size==f['bytes'];z=read(p);j=index[z['job_id']];assert z['GT_read']is False and z['source_case_sha256']==j['source_case_sha256'] and z['prompt_sha256']==j['prompt_sha256'];raw=re.sub(r'^```(?:json)?\s*','',z['raw_output'].strip());raw=re.sub(r'\s*```$','',raw)
  try:
   a=json.loads(raw);v=a['id'];assert type(v)is int and v in j['allowed_output_ids'];valid=True;selected=j['visible_id_to_candidate'].get(str(v))
  except Exception:valid=False;v=None;selected=None
  assert (z['parsed']['status']=='valid')==valid and z['parsed'].get('selected_candidate_id')==selected
  rows.append(dict(case_id=z['case_id'],dataset=z['dataset'],cohort=j['cohort'],kind=z['kind'],format_valid=valid,visible_id=v,selected_candidate_id=selected,truncated=z['output_truncated_at_cap'],candidate_count=j['candidate_count'],input_tokens=z['input_tokens'],visual_tokens=z['visual_tokens'],output_tokens=z['output_tokens'],end_to_end_request_seconds=z['end_to_end_request_seconds'],model_generate_seconds=z['model_forward_generate_seconds']))
  for k in ['input_tokens','visual_tokens','output_tokens','end_to_end_request_seconds']:cost[k]+=z[k]
  cost['model_generate_seconds']+=z['model_forward_generate_seconds'];cost['GPU_peak_bytes']=max(cost['GPU_peak_bytes'],z['GPU_peak_bytes'])
 cases=[];refs=[]
 for inp in read(B/'CANDIDATE_INPUT_BARRIER.json')['records']:
  r={z['kind']:z for z in rows if z['case_id']==inp['case_id']};selected=r['base']['selected_candidate_id'];candidate=next((x for x in inp['candidates']if x['candidate_id']==selected),None);no_update=candidate is None or candidate['is_Source'];anchors=[]
  if not no_update:
   for p,f,b in zip(inp['positions'],inp['physical_frame_ids'],candidate['boxes_xyxy']):
    if b is None:continue
    x1,y1,x2,y2=b;anchors.append(dict(position=p,physical_frame_id=f,box_cxcywh=[(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1]))
   no_update=not anchors
  refs.append(dict(case_id=inp['case_id'],dataset=inp['dataset'],cohort=inp['cohort'],source_case_sha256=inp['source_case_sha256'],references=dict(teacher=dict(no_update=no_update,reason='Source_or_none_invalid_or_empty' if no_update else 'nonSource_base_selected',selected_candidate_id=selected,anchors=anchors))))
  checks={k+'_same_geometry':r['base']['format_valid'] and r[k]['format_valid'] and selected==r[k]['selected_candidate_id']for k in ['id_swap','color_swap','order_swap','joint_swap']}
  cases.append(dict(case_id=inp['case_id'],dataset=inp['dataset'],cohort=inp['cohort'],base_selected_candidate_id=selected,base_is_Source=bool(candidate and candidate['is_Source']),no_update=no_update,actual_reference_anchor_count=len(anchors),**checks,wrong_query_choice_changed=selected!=r['wrong_query']['selected_candidate_id']))
 write(B/'ROOT_TEACHER_READBACK.json',dict(status='pass',count=96,rows=rows,cases=cases,cost=cost,GT_read=False,format_valid_count=sum(r['format_valid']for r in rows),truncated_count=sum(r['truncated']for r in rows),not_identity_qualification=True))
 write(B/'METHOD_REFERENCE_LOCK.json',dict(status='locked',phase='method',records=refs,count=16,GT_read=False,teacher_barrier_sha256=sha(B/'TEACHER_GLOBAL_BARRIER.json'),expected_fit_count=3*sum(not r['references']['teacher']['no_update']for r in refs),time=time.time()))
 print(dict(valid=sum(r['format_valid']for r in rows),no_update=sum(c['no_update']for c in cases),selected=[(c['case_id'],c['base_selected_candidate_id'],c['actual_reference_anchor_count'])for c in cases]))
if __name__=='__main__':main()
