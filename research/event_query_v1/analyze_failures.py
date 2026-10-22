"""Post-seal offline diagnosis of interval misranking, never inference feedback."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).parent));from run import R,D,P,read,put,sha,records
from stvg.metrics import DenseTube
def main():
    seal=read(D/'sealed.json');score=read(D/'score_receipt.json');assert score['seal_sha256']==sha(D/'sealed.json');report=read(P/'report.json');assert sha(P/'report.json')==score['report_sha256'];by={r['query']:r for r in report['rows']}
    label=R/'data/evaluation/development_labels.jsonl';assert sha(label)==score['labels_sha256'];labels={r['id']:r for r in map(json.loads,label.read_text().splitlines())if r['dataset']=='hc2'};cases=[];groups={}
    for row in records():
        key=row['id'];e=seal['files'][key];assert sha(D/e['path'])==e['sha256']and sha(D/e['trace_path'])==e['trace_sha256'];tr=torch.load(D/e['trace_path'],map_location='cpu',weights_only=False);g=labels[key];vals=by[key]['fixed_space'];m=row['metadata'];ids=np.asarray(m['frame_ids']);pairs=tr['feedback']['pairs'];spans=np.stack([ids[pairs[:,0]],ids[pairs[:,1]]+1],-1);gt=np.array(g['interval']);overlap=np.maximum(0,np.minimum(spans[:,1],gt[1])-np.maximum(spans[:,0],gt[0]));t=overlap/(spans[:,1]-spans[:,0]+gt[1]-gt[0]-overlap)
        with np.load(D/e['path'],allow_pickle=False)as z:positions=z['positions8'];actual={a:z[a+'_interval']for a in ['NativeG6','RawDirect','RawQuery','PredictiveDirect','PredictiveQuery']}
        anchors=ids[positions]+.5;pattern=(anchors[None]>=spans[:,:1])&(anchors[None]<spans[:,1:]);codes=pattern@(1<<np.arange(8));per={}
        for kind,f in [('Raw','raw'),('Predictive','predictive')]:
            s=tr['feedback'][f];best=int(np.argmax(s));qa=actual[kind+'Query'];da=actual[kind+'Direct'];selected=int(np.flatnonzero((spans==da).all(1))[0]);same=codes==codes[selected];base_t=vals['NativeG6']['t'];q_t=vals[kind+'Query']['t'];d_t=vals[kind+'Direct']['t']
            per[kind]=dict(evidence_argmax_t=float(t[best]),evidence_argmax_score=float(s[best]),direct_selected_t=d_t,query_selected_t=q_t,query_minus_native_t_pp=100*(q_t-base_t),direct_minus_native_t_pp=100*(d_t-base_t),best_t_same_anchor_pattern=float(t[same].max()),worst_t_same_anchor_pattern=float(t[same].min()),same_pattern_candidates=int(same.sum()),evidence_score_spread=float(np.ptp(s)),loss_decreased=by[key]['quality'][kind]['loss_last']<by[key]['quality'][kind]['loss_first']-1e-10,query_expected_v_delta_pp=100*(by[key]['quality'][kind]['query_expected_fixed_v']-by[key]['quality'][kind]['native_expected_fixed_v']))
        duration=(gt[1]-gt[0])/m['fps'];group='short_le5s'if duration<=5 else'long_gt5s';entry=dict(query=key,duration_group=group,inside_observed=int(((anchors>=gt[0])&(anchors<gt[1])).sum()),candidate_oracle_t=float(t.max()),native_t=vals['NativeG6']['t'],feedback_unknown=by[key]['feedback'].get('unknown'),diagnosis=per);cases.append(entry);groups.setdefault(group,[]).append(entry)
    summary={}
    for group,rs in dict(all=cases,**groups).items():
        summary[group]={kind:dict(queries=len(rs),direct_harms=sum(r['diagnosis'][kind]['direct_minus_native_t_pp']< -1e-9 for r in rs),query_harms=sum(r['diagnosis'][kind]['query_minus_native_t_pp']< -1e-9 for r in rs),query_t_delta_pp=float(np.mean([r['diagnosis'][kind]['query_minus_native_t_pp']for r in rs])),direct_t_delta_pp=float(np.mean([r['diagnosis'][kind]['direct_minus_native_t_pp']for r in rs])),lower_loss_but_harmed=sum(r['diagnosis'][kind]['loss_decreased']and r['diagnosis'][kind]['query_minus_native_t_pp']< -1e-9 for r in rs),direct_harm_with_better_same_evidence_pattern=sum(r['diagnosis'][kind]['direct_minus_native_t_pp']< -1e-9 and r['diagnosis'][kind]['best_t_same_anchor_pattern']>r['native_t']+1e-10 for r in rs))for kind in ['Raw','Predictive']}
    out=dict(scope='Post-seal offline diagnostic; GT never used by targets/query/selection',summary=summary,rows=cases,limitations=['same pattern is only eight observed membership, not equality of physical contrast score','LOO observations are temporally fused and not independent event labels','short/long grouping is offline only; no duration-conditioned inference','GT candidate maximum is diagnostic, not an output or performance guarantee'],report_sha256=sha(P/'report.json'))
    put(P/'failure_analysis.json',out);print(json.dumps(summary))
if __name__=='__main__':main()
