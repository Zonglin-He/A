"""Separately bound POSTSEAL restoration-placement audit; never an online gate."""
import sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.native_event_caption_restore_v1.run import D,P,read,put,sha
import torch,numpy as np
def main():
 torch.set_num_threads(1);s=read(D/'sealed.json');r=read(P/'report.json');assert s['count']==64 and not s['GT_read'];assert read(D/'score_receipt.json')['report_sha256']==sha(P/'report.json')
 oldfile=R/'results/native_event_support_view_hc64_v1/support_audit.json';old=read(oldfile);oldrows={x['query']:x for x in old['rows']};vals={x['query']:x for x in r['rows']};rows=[]
 for k,e in s['files'].items():
  assert sha(D/e['trace_path'])==e['trace_sha256'];tr=torch.load(D/e['trace_path'],weights_only=False,map_location='cpu');stats=[x for off in tr['restoration_stats']for x in off];q=tr['q'];pairs={a:float(np.abs(q['CaptionRestorationFusion']-q[a]).sum()/2)for a in ['DenseFillFusion','UniformRestorationFusion','SpatialPermutationFusion','TemporalRankFusion']}
  mass=[x['mean_restoration']for x in stats if x['mean_restoration']is not None]
  rows.append(dict(query=k,caption_tokens=tr['caption_token_count'],mean_outside_original_restoration=float(np.mean(mass))if mass else None,zero_attention_frames=sum(x['zero_fallback']for x in stats),mean_spatial_permutation_L1=float(np.mean([x['semantic_spatial_L1']for x in stats])),mean_time_rank_L1=float(np.mean([x['semantic_temporal_L1']for x in stats])),posterior_TV=pairs,values=vals[k]['values'],source_duration_group=vals[k]['duration_group'],stage17_target_support_audit=oldrows[k]))
 out=dict(scope='Postseal scored development diagnostics only. No GT selection, new inference or causal attention claim. Previous targetcoverage is only target proxy, not interaction-object GT.',queries=64,mean_outside_original_restoration=float(np.mean([x['mean_outside_original_restoration']for x in rows if x['mean_outside_original_restoration']is not None])),zero_attention_frames=sum(x['zero_attention_frames']for x in rows),mean_posterior_TV={a:float(np.mean([x['posterior_TV'][a]for x in rows]))for a in rows[0]['posterior_TV']},rows=rows)
 put(P/'restoration_audit.json',out);put(D/'restoration_audit_receipt.json',dict(audit_code_sha256=sha(Path(__file__)),original_seal_sha256=sha(D/'sealed.json'),scored_report_sha256=sha(P/'report.json'),parent_support_audit_sha256=sha(oldfile),output_sha256=sha(P/'restoration_audit.json'),postseal_only=True,new_model_calls=0,new_labels_access=0))
 print(json.dumps({k:v for k,v in out.items()if k!='rows'},indent=2))
if __name__=='__main__':main()
