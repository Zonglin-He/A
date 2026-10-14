"""Independent sealed scene-ID selection/cost readback, no GT loader."""
import sys,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_temporal_caption_query_v1 import read,write,sha
B=ROOT/'artifacts/stvg_scene_object_id_referent_v1';bar=read(B/'TEACHER_GLOBAL_BARRIER.json');rt=read(B/'RUNTIME.json');jobs=read(B/'TEACHER_JOB_LOCK.json')['records'];assert len(jobs)==bar['count']==40
for p,h in {**rt['pins'],**rt['protected']}.items():assert sha(ROOT/p)==h
index={j['job_id']:j for j in jobs};rows=[];cost=dict(input_tokens=0,visual_tokens=0,output_tokens=0,end_to_end_request_seconds=0.,model_generate_seconds=0.,GPU_peak_bytes=0)
for f in bar['files']:
 assert sha(ROOT/f['path'])==f['sha256'];z=read(ROOT/f['path']);j=index[z['job_id']];assert z['source_case_sha256']==j['source_case_sha256'] and z['prompt_sha256']==j['prompt_sha256']and z['GT_read']is False;rows.append(dict(case_id=z['case_id'],dataset=z['dataset'],kind=z['kind'],format_valid=z['parsed']['status']=='valid',truncated=z['output_truncated_at_cap'],visible_id=z['parsed'].get('visible_id'),selected_candidate_id=z['parsed'].get('selected_candidate_id'),candidate_count=j['candidate_count'],source_only_non_comparable=j['source_only_non_comparable'],empty_candidate_pool=j['empty_candidate_pool'],input_tokens=z['input_tokens'],visual_tokens=z['visual_tokens'],output_tokens=z['output_tokens'],end_to_end_request_seconds=z['end_to_end_request_seconds'],model_generate_seconds=z['model_forward_generate_seconds']))
 for k in ['input_tokens','visual_tokens','output_tokens','end_to_end_request_seconds']:cost[k]+=z[k]
 cost['model_generate_seconds']+=z['model_forward_generate_seconds'];cost['GPU_peak_bytes']=max(cost['GPU_peak_bytes'],z['GPU_peak_bytes'])
cases=[]
for inp in read(B/'CANDIDATE_INPUT_BARRIER.json')['records']:
 r={z['kind']:z for z in rows if z['case_id']==inp['case_id']};cases.append(dict(case_id=inp['case_id'],dataset=inp['dataset'],base_selected_candidate_id=r['base']['selected_candidate_id'],ID_swap_same_geometry=r['base']['selected_candidate_id']==r['id_swap']['selected_candidate_id'],wrong_query_choice_changed=r['base']['selected_candidate_id']!=r['wrong_query']['selected_candidate_id'],Source_removed_selected_candidate_id=r['Source_removed']['selected_candidate_id'],unmarked_none_returns_zero=r['unmarked_none']['visible_id']==0,candidate_count=inp['selected_candidate_count'],source_only_non_comparable=inp['source_only_non_comparable']))
write(B/'ROOT_TEACHER_READBACK.json',dict(status='pass',count=40,rows=rows,cases=cases,format_valid_count=sum(r['format_valid']for r in rows),truncated_count=sum(r['truncated']for r in rows),ID_swap_same_geometry_count=sum(r['ID_swap_same_geometry']for r in cases),wrong_query_choice_changed_count=sum(r['wrong_query_choice_changed']for r in cases),unmarked_none_zero_count=sum(r['unmarked_none_returns_zero']for r in cases),cost=cost,Source_removed_not_claimed_wrong_object_control=True,GT_read=False))
print(dict(valid=sum(r['format_valid']for r in rows),swaps=sum(r['ID_swap_same_geometry']for r in cases),wrongquerychange=sum(r['wrong_query_choice_changed']for r in cases)))
