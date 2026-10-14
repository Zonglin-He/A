"""Root independent sealed teacher custody and strict/native-ID diagnostic."""
import sys,json,re,hashlib,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_temporal_caption_query_v1 import read,write,sha
B=ROOT/'artifacts/stvg_video_conditioned_global_referent_v1';bar=read(B/'TEACHER_GLOBAL_BARRIER.json');rt=read(B/'RUNTIME.json');jobs=read(B/'TEACHER_JOB_LOCK.json')['records'];assert len(jobs)==bar['count']==42 and bar['runtime_sha256']==sha(B/'RUNTIME.json')
for p,h in {**rt['pins'],**rt['protected']}.items():assert sha(ROOT/p)==h,p
by={j['job_id']:j for j in jobs};out=[];totalin=totalout=totalvisual=0;seconds=modelsec=0.;peak=0
for f in bar['files']:
 assert sha(ROOT/f['path'])==f['sha256'];z=read(ROOT/f['path']);j=by[z['job_id']];assert z['source_case_sha256']==j['source_case_sha256'] and z['GT_read']is False and z['prompt_sha256']==j['prompt_sha256'];mapping={v['visible_id']:v['candidate_id']for v in j['videos']};raw=z['raw_output'].strip();raw=re.sub(r'^```(?:json)?\s*','',raw);raw=re.sub(r'\s*```$','',raw)
 try:
  native=json.loads(raw);ident=native['id'];assert isinstance(ident,int)and not isinstance(ident,bool)and (ident in mapping or ident==0);choice=mapping.get(ident);valid=True
 except Exception:valid=False;ident=None;choice=None
 out.append(dict(case_id=j['case_id'],dataset=j['dataset'],kind=j['kind'],strict_format_valid=z['parsed']['status']=='valid',native_ID_only_valid=valid,native_visible_id=ident,native_selected_candidate_id=choice,strict_selected_candidate_id=z['parsed'].get('selected_candidate_id'),individual_verdict=z['parsed'].get('verdict'),candidate_count=len(j['videos']),truncated=z['output_truncated_at_cap'],input_tokens=z['input_tokens'],output_tokens=z['output_tokens'],visual_tokens=z['visual_tokens'],end_to_end_request_seconds=z['end_to_end_request_seconds'],model_seconds=z['model_forward_generate_seconds']))
 totalin+=z['input_tokens'];totalout+=z['output_tokens'];totalvisual+=z['visual_tokens'];seconds+=z['end_to_end_request_seconds'];modelsec+=z['model_forward_generate_seconds'];peak=max(peak,z['GPU_peak_bytes'])
caseout=[]
for inp in read(B/'CANDIDATE_INPUT_BARRIER.json')['records']:
 cid=inp['case_id'];r={x['kind']:x for x in out if x['case_id']==cid and x['kind']!='individual'};m=read(B/'METHOD_REFERENCE_LOCK.json')['records'];ref=next(x for x in m if x['case_id']==cid)['references']['teacher'];caseout.append(dict(case_id=cid,dataset=inp['dataset'],candidate_count=len(inp['candidates']),ranking_comparison_possible=len(inp['candidates'])>1,base=r['base'],id_swap=r['id_swap'],wrong_query=r['wrong_query'],native_ID_geometric_swap_consistency=(r['base']['native_ID_only_valid'] and r['id_swap']['native_ID_only_valid'] and r['base']['native_selected_candidate_id']==r['id_swap']['native_selected_candidate_id']),wrong_query_preference_changed=r['base']['native_selected_candidate_id']!=r['wrong_query']['native_selected_candidate_id'],method_selected_candidate_id=ref['candidate_id'],method_source_fallback=ref['Source_fallback'],strict_format_vs_native_ID_scope_separate=True))
write(B/'ROOT_TEACHER_READBACK.json',dict(status='pass',count=42,case_count=8,all_sealed_sha_source_prompt_checked=True,rows=out,cases=caseout,strict_valid_count=sum(x['strict_format_valid']for x in out),native_ID_only_valid_count=sum(x['native_ID_only_valid']for x in out),truncated_count=sum(x['truncated']for x in out),cost=dict(input_tokens=totalin,visual_tokens=totalvisual,output_tokens=totalout,end_to_end_requests_seconds=seconds,model_generate_seconds=modelsec,GPU_peak_bytes=peak),native_parser_diagnostic_only_no_method_reselection=True,GT_read=False))
print(dict(strict=sum(x['strict_format_valid']for x in out),nativeID=sum(x['native_ID_only_valid']for x in out),swaps=sum(x['native_ID_geometric_swap_consistency']for x in caseout),changed_query=sum(x['wrong_query_preference_changed']for x in caseout),cost_seconds=seconds))
