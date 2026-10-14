"""Independent response parsing and motion-evidence reference locking before GT."""
import sys,json,re,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.stvg_temporal_caption_query_v1 import read,write,sha,catalog,inference_guard
B=ROOT/'artifacts/stvg_adjacent_motion_opd_v1'
def evidence(track):
 transitions={r['to_physical_frame_id']:r for r in track['transitions'] if r['status']=='accepted'}
 out=[]
 for r in track['trajectory']:
  box=r['box_xyxy_norm'];fid=r['physical_frame_id']
  rho=1. if r['status']=='anchor' else float(transitions[fid]['rho']) if box is not None else 0.
  out.append(dict(physical_frame_id=fid,center_xy=[(box[0]+box[2])/2,(box[1]+box[3])/2] if box is not None else None,rho=rho))
 return out

def main():
 sys.addaudithook(inference_guard)
 bar=read(B/'TEACHER_GLOBAL_BARRIER.json');rt=read(B/'RUNTIME.json');jobs=read(B/'TEACHER_JOB_LOCK.json')['records'];assert bar['count']==len(jobs)==48 and bar['GT_read'] is False
 assert bar['runtime_sha256']==sha(B/'RUNTIME.json') and bar['input_barrier_sha256']==sha(B/'CANDIDATE_INPUT_BARRIER.json') and bar['job_lock_sha256']==sha(B/'TEACHER_JOB_LOCK.json')
 for p,h in {**rt['pins'],**rt['protected']}.items():assert sha(ROOT/p)==h,p
 index={j['job_id']:j for j in jobs};rows=[];seen=set();cost=dict(input_tokens=0,visual_tokens=0,output_tokens=0,end_to_end_request_seconds=0.,model_generate_seconds=0.,GPU_peak_bytes=0)
 for f in bar['files']:
  p=ROOT/f['path'];assert sha(p)==f['sha256'] and p.stat().st_size==f['bytes'];z=read(p);assert z['job_id'] not in seen;seen.add(z['job_id']);j=index[z['job_id']];assert z['GT_read']is False and z['source_case_sha256']==j['source_case_sha256'] and z['prompt_sha256']==j['prompt_sha256']
  for v in j['videos']:assert sha(ROOT/v['path'])==v['sha256']
  assert z['input_videos']==j['videos'] and z['candidate_bindings']==j['candidate_bindings']
  raw=re.sub(r'^```(?:json)?\s*','',z['raw_output'].strip());raw=re.sub(r'\s*```$','',raw)
  try:
   a=json.loads(raw);v=a['id'];assert type(v)is int and v in j['allowed_output_ids'];valid=True;selected=j['visible_id_to_candidate'].get(str(v))
  except Exception:valid=False;v=None;selected=None
  assert (z['parsed']['status']=='valid')==valid and z['parsed'].get('selected_candidate_id')==selected
  rows.append(dict(case_id=z['case_id'],dataset=z['dataset'],cohort=j['cohort'],kind=z['kind'],format_valid=valid,visible_id=v,selected_candidate_id=selected,truncated=z['output_truncated_at_cap'],candidate_count=j['candidate_count'],input_tokens=z['input_tokens'],visual_tokens=z['visual_tokens'],output_tokens=z['output_tokens'],end_to_end_request_seconds=z['end_to_end_request_seconds'],model_generate_seconds=z['model_forward_generate_seconds']))
  for k in ['input_tokens','visual_tokens','output_tokens','end_to_end_request_seconds']:cost[k]+=z[k]
  cost['model_generate_seconds']+=z['model_forward_generate_seconds'];cost['GPU_peak_bytes']=max(cost['GPU_peak_bytes'],z['GPU_peak_bytes'])
 assert seen==set(index)
 roster={r['case_id']:r for r in read(B/'ROSTER_LOCK.json')['records']};cases=[];refs=[]
 for inp in read(B/'CANDIDATE_INPUT_BARRIER.json')['records']:
  c=catalog(roster[inp['case_id']])[0];grid=list(map(int,c['frame_ids']));source=c['zero']['boxes'].cpu().double().numpy()
  r={z['kind']:z for z in rows if z['case_id']==inp['case_id']};selected=r['base']['selected_candidate_id'];candidate=next((x for x in inp['candidate_geometry'] if x['candidate_id']==selected),None)
  anchors=[];ev=[];paircount=0;delta=0.;support=0
  if candidate is not None and candidate.get('tracklet',{}).get('trajectory'):
   track=candidate['tracklet'];ev=evidence(track);byfid={x['physical_frame_id']:x for x in track['trajectory']}
   for p,f in zip(inp['positions'],inp['physical_frame_ids']):
    b=byfid[f]['box_xyxy_norm']
    if b is None:continue
    x1,y1,x2,y2=b;anchors.append(dict(position=p,physical_frame_id=f,box_cxcywh=[(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1]))
   support=len(anchors)
   for left,right in zip(ev,ev[1:]):
    if left['center_xy'] is not None and right['center_xy'] is not None and min(left['rho'],right['rho'])>0 and right['physical_frame_id']==left['physical_frame_id']+1 and grid[0]<=left['physical_frame_id'] and right['physical_frame_id']<=grid[-1]:paircount+=1
   for item in track['trajectory']:
    b=item['box_xyxy_norm'];f=item['physical_frame_id']
    if b is None or not grid[0]<=f<=grid[-1]:continue
    x1,y1,x2,y2=b;geo=np.array([(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1]);old=np.array([np.interp(f,grid,source[:,j]) for j in range(4)]);delta=max(delta,float(np.abs(geo-old).max()))
  eligible=support>=2 and paircount>=1 and delta>1e-6 and r['base']['format_valid'] and not r['base']['truncated']
  reasons=[]
  if candidate is None:reasons.append('abstain_or_invalid')
  if support<2:reasons.append('fewer_than_two_observation_support')
  if paircount<1:reasons.append('no_consecutive_reliable_pair_in_source_grid')
  if delta<=1e-6:reasons.append('no_independent_geometric_correction')
  if r['base']['truncated']:reasons.append('truncated_base_response')
  teacher=dict(no_update=not eligible,eligible_for_update=eligible,reason='independent_motion_evidence' if eligible else '|'.join(reasons),selected_candidate_id=selected,is_Source=bool(candidate and candidate['is_Source']),anchors=anchors,motion_evidence=ev,evidence_observation_count=support,evidence_pair_count=paircount,maximum_normalized_correction=delta,geometry_source='independent_adjacent_similarity_affine_track',source_display_is_original_geometry=bool(candidate and candidate['is_Source']))
  refs.append(dict(case_id=inp['case_id'],dataset=inp['dataset'],cohort=inp['cohort'],source_case_sha256=inp['source_case_sha256'],references=dict(teacher=teacher)))
  checks={k+'_same_geometry':r['base']['format_valid'] and r[k]['format_valid'] and selected==r[k]['selected_candidate_id'] for k in ['id_swap','color_swap','order_swap','joint_swap']}
  cases.append(dict(case_id=inp['case_id'],dataset=inp['dataset'],cohort=inp['cohort'],base_selected_candidate_id=selected,base_is_Source=teacher['is_Source'],no_update=not eligible,eligible_for_update=eligible,actual_reference_anchor_count=support,evidence_pair_count=paircount,maximum_normalized_correction=delta,**checks,wrong_query_choice_changed=selected!=r['wrong_query']['selected_candidate_id']))
 write(B/'ROOT_TEACHER_READBACK.json',dict(status='pass',count=48,rows=rows,cases=cases,cost=cost,GT_read=False,format_valid_count=sum(r['format_valid'] for r in rows),truncated_count=sum(r['truncated'] for r in rows),not_identity_qualification=True))
 write(B/'METHOD_REFERENCE_LOCK.json',dict(status='locked',phase='method',records=refs,count=8,GT_read=False,teacher_barrier_sha256=sha(B/'TEACHER_GLOBAL_BARRIER.json'),expected_fit_count=3*sum(not r['references']['teacher']['no_update'] for r in refs),time=time.time()))
 print(dict(valid=sum(r['format_valid'] for r in rows),no_update=sum(c['no_update'] for c in cases),cases=cases))
if __name__=='__main__':main()
