"""Sealed-source, three-seed-averaged evaluation; never imported by training workers."""
import argparse,json,re,sys,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.decota_matrix_common_v1 import read,write,status,load,sha
from scripts.ptd_joint_box_opd_v1 import OUT,rows,file,verify,SEEDS,checked
from scripts.score_ptd_spatial_adapter_ab_v1 import truth,evaluate

def stat(vals):
    v=np.array([v for v in vals if v is not None],float)
    if not len(v):return dict(n=0,mean=None,ci95=None)
    rng=np.random.default_rng(20260925);boot=v[rng.integers(0,len(v),(10000,len(v)))].mean(1)
    return dict(n=len(v),mean=float(v.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),positive=int((v>0).sum()),negative=int((v<0).sum()),zero=int((v==0).sum()),
        severe_loss_gt5pp=int((v<-.05).sum()),minimum=float(v.min()),maximum=float(v.max()),median=float(np.median(v)))
def signflip(v):
    x=np.asarray(v,float);observed=abs(x.mean());rng=np.random.default_rng(20260925);count=0
    for _ in range(100):
        signs=rng.integers(0,2,(1000,len(x)))*2-1
        count+=int((np.abs((signs*x).mean(1))>=observed-1e-15).sum())
    return (count+1)/100001

def main(stage):
    torch.set_num_threads(4);verify();bar=read(OUT/f'PREDICTION_BARRIER_{stage}.json')
    for p,h in bar['files'].items():assert sha(p)==h,p
    intake=read(OUT/'INTAKE_COMPLETE.json');assert sha(OUT/'LABELS_SCORER_ONLY.json')==intake['labels_sha'];labels=read(OUT/'LABELS_SCORER_ONLY.json')
    rr=rows(stage);assert len(rr)==(64 if stage=='development' else 192)
    arms=['D0']+bar['arms'];results=[];diag=[]
    for row in rr:
        native=checked(file(row,'native'));g=truth(row,labels);rec=dict(key=row['key'],source=row['source'],domain=row['domain'],cohort=row['cohort'],official_split=row['official_split'],arms={},seeds={})
        for arm in arms:
            metrics=[]
            for seed in ([None] if arm=='D0' else SEEDS):
                p=native if arm=='D0' else checked(file(row,'predictions',arm,seed))
                parsed=re.findall(r'<t(\d+)><\|box_start\|><(\d+)><(\d+)><(\d+)><(\d+)><\|box_end\|>',p['completion'])
                tokens=torch.tensor([[int(v) for v in x[1:]] for x in parsed])
                if p['format_ok']:
                    assert [int(x[0])-1 for x in parsed]==p['positions']
                    assert torch.equal(tokens,p.get('adapted_logits',p['logits']).argmax(-1))
                v=evaluate(p,tokens,row,g);metrics.append(v)
                if seed is not None:rec['seeds'][f'{arm}_{seed}']={k:v[k] for k in ['v','s','t','invalid_boxes','format_ok']}
            rec['arms'][arm]={k:float(np.mean([m[k] for m in metrics if m[k] is not None])) if any(m[k] is not None for m in metrics) else None for k in ['v','s','t','coverage','invalid_boxes']}
            rec['arms'][arm].update(ious=np.mean([m['ious'] for m in metrics],0).tolist(),common_positions=metrics[0]['common_positions'],format_ok=all(m['format_ok'] for m in metrics))
        ids=np.asarray(row['input']['frame_ids']);present=np.isin(np.arange(32),native.get('positions',[]));legal=present&g['valid']
        if native['interval'] is not None:
            a,b=native['interval'];legal &= (ids>=ids[a])&(ids<ids[b]+1)
        else:legal[:]=False
        base=np.asarray(rec['arms']['D0']['ious']);good=legal&(base>=.5);bad=legal&~good
        rec['support']=dict(legal=int(legal.sum()),native_good_frames=int(good.sum()),native_bad_frames=int(bad.sum()),GT_valid=int(g['valid'].sum()))
        rec['retention']={a:dict(good=int(good.sum()),retained=int((good&(np.asarray(rec['arms'][a]['ious'])>=.5)).sum()),
            continuous_harmed=int((good&(np.asarray(rec['arms'][a]['ious'])<base)).sum()),harmed_gt5pp=int((good&(np.asarray(rec['arms'][a]['ious'])-base<-.05)).sum()),
            bad_rescued=int((bad&(np.asarray(rec['arms'][a]['ious'])>=.5)).sum()),
            bad_delta=float((np.asarray(rec['arms'][a]['ious'])-base)[bad].mean()) if bad.any() else None) for a in arms if a!='D0'}
        areas=g['boxes'][:,2:].prod(-1);rec['groups']=dict(native_good=rec['arms']['D0']['s'] is not None and rec['arms']['D0']['s']>=.5,
            native_bad=rec['arms']['D0']['s'] is not None and rec['arms']['D0']['s']<.5,time_coverage_low=rec['arms']['D0']['coverage']<.5,
            small_GT=bool(g['valid'].any() and np.median(areas[g['valid']])<=.01),
            relation_query_proxy=bool(re.search(r'\b(left|right|behind|front|beside|between|next|holding|near)\b',row['input']['caption'],re.I)))
        # Joint preference/IoU association uses actual sampled complete boxes, never an update gate.
        if 'J2' in arms and native['format_ok']:
            from scripts.score_ptd_spatial_adapter_ab_v1 import iou
            from vg_tta.ptd_spatial_adapter_ab_v1 import boxes_from_tokens
            for seed in SEEDS:
                fit=checked(file(row,'fits','J2',seed))
                for step,h in enumerate(fit['history'][:3]):
                    actions=h['actions'];boxes,_=boxes_from_tokens(actions);quality=iou(boxes.numpy(),g['boxes'][np.asarray(native['positions'])][:,None])
                    scores=h['teacher_segments'].sum(-1).numpy()
                    for j,pos in enumerate(native['positions']):
                        if not legal[pos]:continue
                        diag.append(dict(key=row['key'],seed=seed,step=step,position=pos,scores=scores[j].tolist(),ious=quality[j].tolist(),actions=actions[j].tolist()))
        results.append(rec)
    summary={}
    for name,pred in [('all',lambda r:True),('HC',lambda r:r['domain']=='HC'),('Vid',lambda r:r['domain']=='Vid'),('HC_test',lambda r:r['cohort']=='hcstvg1_test'),('HC_train',lambda r:r['cohort']=='hcstvg1_train')]:
        subset=[r for r in results if pred(r)]
        if not subset:continue
        summary[name]={}
        for arm in arms:
            summary[name][arm]=dict(absolute={m:stat([r['arms'][arm][m] for r in subset]) for m in ['v','s','t']},
                dv=stat([r['arms'][arm]['v']-r['arms']['D0']['v'] for r in subset]),
                ds=stat([r['arms'][arm]['s']-r['arms']['D0']['s'] for r in subset if r['arms'][arm]['s'] is not None]),
                vs_D1=stat([r['arms'][arm]['v']-r['arms']['D1']['v'] for r in subset]),
                retention={k:sum(r['retention'][arm][k] for r in subset) for k in ['good','retained','continuous_harmed','harmed_gt5pp','bad_rescued']} if arm!='D0' else None,
                bad_delta=stat([r['retention'][arm]['bad_delta'] for r in subset]) if arm!='D0' else None)
    pairs=[('J2','J1'),('J2','D1')] if stage=='development' and 'J2' in arms else [('D1','D0')]+([('J2','D1')] if 'J2' in arms else [])
    tests={}
    for a,b in pairs:
        vals=[r['arms'][a]['v']-r['arms'][b]['v'] for r in results];tests[a+'-'+b]=dict(effect=stat(vals),p_signflip=signflip(vals))
    ordered=sorted(tests,key=lambda name:tests[name]['p_signflip']);adjusted=0
    for rank,name in enumerate(ordered):
        adjusted=max(adjusted,min(1.,tests[name]['p_signflip']*(len(ordered)-rank)));tests[name]['p_Holm']=adjusted
    group_stats={}
    for group in results[0]['groups']:
        subset=[r for r in results if r['groups'][group]]
        group_stats[group]={a:dict(n=len(subset),dv=stat([r['arms'][a]['v']-r['arms']['D0']['v'] for r in subset])) for a in arms if a!='D0'}
    dest=OUT/stage;write(dest/'SOURCE_RESULTS.json',results);write(dest/'JOINT_GEOMETRY_ASSOCIATION.json',diag)
    write(dest/'SUMMARY.json',dict(summary=summary,tests=tests,groups=group_stats,source_count=len(rr),seeds=SEEDS,
        seed_aggregation='within source mean before source macro',GT_read_time=time.time(),legal_frames=sum(r['support']['legal'] for r in results),
        native_format_invalid=sum(not r['arms']['D0']['format_ok'] for r in results),historical_exposure=True,HC_train_is_training_covered=True))
    if stage=='development':
        if 'J2' not in arms:decision=dict(include_J2=False,reason='joint_interface_unavailable',checks={})
        else:
            checks=dict(better_than_J1=tests['J2-J1']['effect']['mean']>0,better_than_D1=tests['J2-D1']['effect']['mean']>0,
                no_domain_CI_entirely_negative=all(summary[d]['J2']['vs_D1']['ci95'][1]>=0 for d in ['HC','Vid']),
                no_extra_severe=summary['all']['J2']['dv']['severe_loss_gt5pp']<=summary['all']['D1']['dv']['severe_loss_gt5pp'],
                native_good_retained=summary['all']['J2']['retention']['retained']>=summary['all']['D1']['retention']['retained'])
            decision=dict(include_J2=all(checks.values()),checks=checks,rule='pre-registered developmental resource decision; no tuning',D1_confirmation_unconditional=True)
        write(OUT/'CANDIDATE_DECISION.json',dict(**decision,time=time.time(),development_summary_sha=sha(dest/'SUMMARY.json'),confirmation_outcomes_opened=False))
    lines=[f'# {stage}: PTD D1 / Joint-Box OPD','',
        f'{len(rr)} independent parents within this split; three seeds averaged within source. All have historical project exposure. HC official train supplementation is training-covered and separately stratified.','',
        '| Stratum | Arm | sources | vIoU % | Delta v pp | 95% CI pp | sIoU % | s denominator |','|---|---|---:|---:|---:|---|---:|---:|']
    for co,aa in summary.items():
        for arm,x in aa.items():
            v=x['absolute']['v'];s=x['absolute']['s'];dv=x['dv'];sv='NA' if s['mean'] is None else f'{100*s["mean"]:.5f}'
            lines.append(f'|{co}|{arm}|{v["n"]}|{100*v["mean"]:.5f}|{100*dv["mean"]:+.5f}|{[round(100*q,5) for q in dv["ci95"]]}|{sv}|{s["n"]}|')
    lines+=['','Primary comparisons with source sign-flip and Holm adjustment:',json.dumps(tests,ensure_ascii=False,indent=2),'',
        'Native-good/bad and time coverage are offline explanation groups, never training filters. Relation-query and small-GT flags are descriptive proxies, not exhaustive multi-instance/occlusion identity annotations. Complete joint sample probabilities include separators/newline; raw sample energy is not a measured normalized joint KL. No source-level online writeback or semantic/time update.',
        '', 'Raw per-source/seed metrics, continuous negative tails and joint geometry association are saved beside this report. Scientific interpretation and finite image inspection require the accompanying final readback; do not infer method success from process completion.']
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n');status(OUT/'STATUS.json',dict(state='scored',stage=stage,time=time.time()))
    print(stage,'SCORED',tests,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['development','confirmation']);main(ap.parse_args().stage)
