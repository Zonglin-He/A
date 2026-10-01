"""Explain temporal ranking failures from immutable expert proposals, CPU only."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.diagnose_tastvg_pipeline_cpu_v1 import BASE,read,write,sha,truth,torch,np

def iou(a,b):
    inter=max(0,min(a[1],b[1])-max(a[0],b[0]));return inter/max(max(a[1],b[1])-min(a[0],b[0]),1e-12)

def run(panel):
    old=ROOT/'artifacts/tastvg_paper48_v1'/panel;out=BASE/panel
    plan=read(old/'PLAN.json');_,spans,_=truth(panel,plan)
    experts=old/'experts' if panel=='P5' else old.parent/'experts';rows=[]
    for r in read(out/'ROWS.json'):
        if not r['expert_scheduled']:continue
        f=old/'online'/r['condition']/r['order']/f"{r['arrival']:05}.pt"
        assert sha(f)==read(f.with_suffix('.json'))['sha256'];x=torch.load(f,map_location='cpu',weights_only=False);td=x['temporal']
        receipt=read(experts/'temporal'/r['condition']/f"{r['parent']:05}.json");cache=experts/receipt['cache'];assert sha(cache)==receipt['cache_sha256'];e=torch.load(cache,map_location='cpu',weights_only=False)
        proposals=e['proposals'];conf=e['proposal_confidence'];intervals=[c['physical_interval'] for c in td['candidates']]
        scores=np.array([[iou(c,t)*w for t,w in zip(proposals,conf)] for c in intervals]);recomputed=scores.max(1) if len(proposals) else np.zeros(len(intervals));np.testing.assert_allclose(recomputed,td['scores'],atol=1e-14,rtol=0)
        selected=td['selected'];oracle=int(np.argmax([c['v'] for c in r['candidate_metrics']]));gt=spans[r['parent']]
        wi=int(np.argmax(scores[selected])) if len(proposals) else None
        native=intervals[0];chosen=intervals[selected]
        old_overlap=max(0,min(native[1],gt[1])-max(native[0],gt[0]));new_overlap=max(0,min(chosen[1],gt[1])-max(chosen[0],gt[0]))
        z={k:r[k] for k in ['parent','condition','order','arrival','slow_v','final_v','slow_t','final_t','candidate_best_v','candidate_best_t']}
        z.update(selected=selected,oracle_v_index=oracle,native_score=float(recomputed[0]),selected_score=float(recomputed[selected]),oracle_v_score=float(recomputed[oracle]),oracle_at_top_tie=bool(abs(recomputed[oracle]-recomputed[selected])<=1e-12),top_tie_count=int((abs(recomputed-recomputed[selected])<=1e-12).sum()),teacher_proposals=len(proposals),best_teacher_tIoU=max([iou(t,gt) for t in proposals],default=0),winning_teacher_tIoU=iou(proposals[wi],gt) if wi is not None else 0,winning_teacher_confidence=float(conf[wi]) if wi is not None else None,GT_temporal_overlap_change=new_overlap-old_overlap,predicted_length_ratio=(chosen[1]-chosen[0])/(native[1]-native[0]),spatial_reason=r['spatial_reason'])
        rows.append(z)
    summary={}
    for group in ['corruption','clean']:
        rr=[r for r in rows if (r['condition']!='clean')==(group=='corruption')]
        subsets={'scheduled':rr,'native_correct_destroyed_v03':[r for r in rr if r['slow_v']>.3 and r['final_v']<=.3],'correct_candidate_missed_v03':[r for r in rr if r['candidate_best_v']>.3 and r['final_v']<=.3],'correct_candidate_missed_t05':[r for r in rr if r['candidate_best_t']>.5 and r['final_t']<=.5]}
        summary[group]={}
        for name,seq in subsets.items():
            summary[group][name]=dict(cells=len(seq),no_proposals=sum(r['teacher_proposals']==0 for r in seq),oracle_top_tie=sum(r['oracle_at_top_tie'] for r in seq),no_teacher_tIoU_above05=sum(r['best_teacher_tIoU']<=.5 for r in seq),winning_teacher_tIoU_above05=sum(r['winning_teacher_tIoU']>.5 for r in seq),good_teacher_exists_but_bad_winner=sum(r['best_teacher_tIoU']>.5 and r['winning_teacher_tIoU']<=.5 for r in seq),GT_temporal_overlap_decreased=sum(r['GT_temporal_overlap_change']<0 for r in seq),interval_length_increased=sum(r['predicted_length_ratio']>1 for r in seq),native_chosen=sum(r['selected']==0 for r in seq))
    write(out/'CRITIC_ROWS.json',rows);write(out/'CRITIC_SUMMARY.json',summary)
    assert not torch.cuda.is_initialized();print(panel,summary['corruption'],flush=True)

if __name__=='__main__':
    for p in ['P1','P5']:run(p)
