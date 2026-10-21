"""One frozen no-training correction readout on already real executed tubes."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from stvg.runtime import sha
from stvg.tube_runner import write
from tools.tubedetr_universal_diagnosis import inputs as parent_inputs
from tools.universal_trial.absolute_confidence import absolute_confidence_scores
PUBLIC=ROOT/'results/tubedetr_universal_confidence';OUT=ROOT/'outputs/tubedetr_universal_confidence';PLAN=PUBLIC/'plan.json'

def run():
    p=json.loads(PLAN.read_text())
    for f,h in p['code_sha256'].items():assert sha(ROOT/f)==h,f
    f=ROOT/'results/tubedetr_universal_diagnosis/final_seal.json';assert sha(f)==p['parent_final_seal_sha256'];s=json.loads(f.read_text());assert s['status']=='complete'
    for name,h in s['files'].items():assert sha(ROOT/name)==h,name
    _,old,cfg,rows=parent_inputs();parent=ROOT/'outputs/tubedetr_universal_diagnosis';seal=json.loads((parent/'sealed.json').read_text());assert sha(parent/'sealed.json')==p['parent_output_seal_sha256'];entries={}
    for q in old['query_ids']:
        e=seal['files'][q];assert sha(parent/e['path'])==e['sha256']
        with np.load(parent/e['path'])as z:
            # diagnostic_scores columns: support,association,-change,without-protection
            scores,exposure=absolute_confidence_scores(z['diagnostic_scores'][:,3],-z['diagnostic_scores'][:,2],z['confidence'],z['frame_ids'],z['Strong_interval'],old['change_weight'])
        assert np.isfinite(scores).all();path=OUT/'scores'/f'{q}.npz';path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();np.savez(path,scores=scores,exposure=np.array(exposure));entries[q]=dict(path=str(path.relative_to(OUT)),sha256=sha(path))
    write(OUT/'sealed.json',dict(status='complete',queries=16,files=entries,plan_sha256=sha(PLAN),labels_read=False,no_new_student_or_teacher_forward=True))
    # All correction scores are immutable before independent offline GT read.
    lp=ROOT/cfg['labels'];assert sha(lp)==cfg['labels_sha256'];labels={v['id']:v for v in map(json.loads,lp.read_text().splitlines())}
    from stvg.metrics import DenseTube
    from tools.layout_trial.diagnostics import paired_ranking,mean
    from tools.tubedetr_opd_direct_diagnosis import cluster_difference
    native=ROOT/'outputs/tubedetr_universal_trial';ns=json.loads((native/'sealed.json').read_text());details=[]
    for q in old['query_ids']:
        e=ns['files'][q];assert sha(native/e['path'])==e['sha256'];t=labels[q];gt={int(k):v for k,v in t['boxes'].items()}
        with np.load(native/e['path'])as a,np.load(OUT/entries[q]['path'])as z:
            r=z['scores'];index=int(r.argmax());meta=dict(frame_ids=a['frame_ids'].tolist(),input=rows[q]['metadata']);metric=lambda b:DenseTube(b,meta,gt,t['interval']).score(a['Strong_interval']);m=[metric(b)for b in a['initial_boxes']]
            details.append(dict(id=q,parent=t['parent'],source=m[0],corrected=m[index],initial=m[a['initial_rewards'].argmax()],selected_source=index==0,exposure=float(z['exposure']),rank=paired_ranking(r,np.array([v['v']for v in m]))))
    result=dict(status='complete_confidence_scale_readout',queries=16,metrics={n:{k:100*mean([r[n][k]for r in details])for k in ['v','s','t']}for n in ['source','initial','corrected']},selected_source=sum(r['selected_source']for r in details),mean_evidence_exposure=mean([r['exposure']for r in details]),reward_pair_accuracy=mean([r['rank']['pair_accuracy']for r in details]),reward_spearman=mean([r['rank']['spearman']for r in details]),difference_corrected_source=cluster_difference([r['corrected']['v']-r['source']['v']for r in details],[r['parent']for r in details]),no_new_forward_or_training=True,no_coefficient_search=True,scores_sealed_before_GT=True,posthoc_exploratory=True,not_OPD_adaptation_or_success_claim=True,confidence_not_a_probability=True)
    write(OUT/'details_private.json',details);write(PUBLIC/'summary.json',result);(PUBLIC/'report.md').write_text('# 弱证据绝对尺度：单次保守修正\n\n仅16条已真实执行的原21候选；对旧无保护奖励乘本query固定时间支持上的八假设平均置信，再保留原0.25改动保护。不搜索系数、未改变任何网络或轨迹，不把置信当概率。全部新分数封存后读GT，仅探索性诊断，不是新OPD适应。\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    write(PUBLIC/'final_seal.json',dict(status='complete',files={str(f.relative_to(ROOT)):sha(f)for f in [PLAN,PUBLIC/'summary.json',PUBLIC/'report.md']},score_seal_sha256=sha(OUT/'sealed.json'),publication_pending=True));write(OUT/'status.json',dict(status='complete',queries=16,no_GPU=True))

if __name__=='__main__':
    try:run()
    except Exception as e:
        write(OUT/'failure.json',dict(error=type(e).__name__,message=str(e)));raise
