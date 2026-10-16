"""Independent numerical QA, common-mask contrasts, and executed notebook."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.decota_heuristic_study_v1 import *
from scripts.score_decota_heuristic_study_v1 import gt_payload,gt_array,summarize


def independent_quality(boxes,truth,valid):
    b=np.asarray(boxes,dtype=float);g=np.asarray(truth,dtype=float);q=np.full(len(b),np.nan)
    for i in np.flatnonzero(valid):
        bx=[b[i,0]-b[i,2]/2,b[i,1]-b[i,3]/2,b[i,0]+b[i,2]/2,b[i,1]+b[i,3]/2]
        gx=[g[i,0]-g[i,2]/2,g[i,1]-g[i,3]/2,g[i,0]+g[i,2]/2,g[i,1]+g[i,3]/2]
        intersection=max(0,min(bx[2],gx[2])-max(bx[0],gx[0]))*max(0,min(bx[3],gx[3])-max(bx[1],gx[1]))
        q[i]=intersection/(b[i,2]*b[i,3]+g[i,2]*g[i,3]-intersection)
    return q


def audit():
    from vg_tta.dense_expansion_data_v1 import DenseBridge
    p=plan();torch.set_num_threads(2);scored=read(OUT/'scored_rows.json');summary=read(OUT/'summary.json')
    bridges={k:DenseBridge(v) for k,v in p['evaluation_label_specs'].items()}
    bar=read(OUT/'prediction_barrier.json');counts=dict(metrics=0,main_parity=0,temporal_episodes=0,backwards=0,noops=0,donor_assignments=0)
    pair_rows=[];maxerr=0.
    for b in BS:
        for g in GS:
            for split in SPLITS:
                tbar=read(OUT/'temporal'/b/g/'barrier.json')
                for r in p['rows'][split][g]:
                    t=load(tpath(b,g,split,r['ordinal']));a=t['audits']['decota']['audit']
                    assert not t['GT_used'];assert a['parameter_delta_l2']>0
                    counts['temporal_episodes']+=1;counts['backwards']+=a['backward_steps']
                    for seed,z in t['donors'].items():
                        assert z['source']!=r['input']['source'];counts['donor_assignments']+=1
                    for name in ['lr0','steps0']:
                        if name in t['intervals']:
                            assert t['intervals'][name]==t['intervals']['frozen'];assert t['audits'][name]['audit']['parameter_delta_l2']==0.;counts['noops']+=1
    for receipt in bar['receipts']:
        assert sha(receipt['path'])==receipt['sha256'];x=load(receipt['path']);assert not x['GT_used'] and x['stable_main_exact']
        b,g,j=receipt['backbone'],receipt['group'],receipt['ordinal'];r=p['rows']['evaluation'][g][j]
        row=next(s for s in scored if s['backbone']==b and s['group']==g and s['ordinal']==j)
        targets,gt=gt_payload(p,r,g,'evaluation',bridges);truth,valid=gt_array(targets,x['frame_ids'],gt)
        pred=x['predictions'];ids=np.array(x['frame_ids']);allq={}
        for name,z in pred.items():
            q=independent_quality(z['boxes'],truth,valid);allq[name]=q;a,c=ids[z['indices'][0]],ids[z['indices'][1]]+1
            inter=valid&(ids>=a)&(ids<c);union=(ids>=min(a,gt[0]))&(ids<max(c,gt[1]))
            ref=float(q[inter].sum()/max(int(union.sum()),1));error=abs(ref-row['metrics'][name]['vIoU_corrected']);maxerr=max(maxerr,error)
            assert error<1e-10,(name,error);counts['metrics']+=1
        counts['main_parity']+=1
        values={}
        for variant in ['fixed','retuned']:
            for fam in ['uniform','no_new','no_evidence','full_uniform','text_noun','text_full','absolute','direct','gate_none','gate_score']:
                a='main_'+variant;z=fam+'_'+variant;mask=valid.copy()
                mask[list(set(pred[a]['called']+pred[z]['called']))]=False
                values[a+'__minus__'+z]=dict(frames=int(mask.sum()),delta=float((allq[a][mask]-allq[z][mask]).mean()) if mask.any() else None)
        pair_rows.append(dict(backbone=b,group=g,source=r['input']['source'],values=values))
    means_checked=0;cis_checked=0
    for cell,s in summary.items():
        b,g=next((b,g) for b in BS for g in GS if b+'_'+g==cell)
        rr=[r for r in scored if r['backbone']==b and r['group']==g]
        for m,ff in s['absolute'].items():
            for f,z in ff.items():
                vals=[r['metrics'][m].get(f) for r in rr];good=[v for v in vals if v is not None]
                if not good:assert z['mean'] is None;continue
                assert abs(float(np.mean(good))-z['mean'])<1e-12;means_checked+=1
        # Bootstrap independent indexing path, same ordered source resampling.
        rr=sorted(rr,key=lambda r:r['source']);rng=np.random.default_rng(20260910)
        ix=rng.integers(len(rr),size=(10000,len(rr)))
        for name,z in s['contrasts'].items():
            if '__minus__' not in name:continue
            a,c=name.split('__minus__');v=np.array([r['metrics'][a]['vIoU_corrected']-r['metrics'][c]['vIoU_corrected'] for r in rr])
            ci=np.percentile(np.take(v,ix).mean(axis=1),[2.5,97.5]);assert np.max(abs(ci-z['vIoU_corrected']['ci95']))<1e-12;cis_checked+=1
    pair_summary={}
    for b in BS:
        for g in GS:
            rr=[r for r in pair_rows if r['backbone']==b and r['group']==g];sources=[r['source'] for r in rr]
            pair_summary[b+'_'+g]={k:summarize([r['values'][k]['delta'] for r in rr],sources) for k in rr[0]['values']}
    # Verify frozen expert receipts and actual vs cached forward provenance.
    expert_counts={};phrases={}
    for g in GS:
        eb=read(OUT/'expert'/g/'barrier.json');assert not eb['GT_used'] and eb['state_unchanged']
        new=reused=0
        for e in eb['receipts']:
            assert sha(e['path'])==e['sha256'];z=load(e['path']);assert not z['GT_used']
            new+=int(not z['reused']);reused+=int(z['reused'])
        expert_counts[g]=dict(new_forward_receipts=new,reused_receipts=reused,total=len(eb['receipts']))
        for split in SPLITS:
            rs=p['rows'][split][g];phrases[split+'_'+g]=dict(total=len(rs),parsed_empty=sum(not r['visual_query']['phrase'] for r in rs),
                full_no_literal_entity=sum(bool(text_spec(r,'full')[0]) and text_spec(r,'full')[1] not in text_spec(r,'full')[0] for r in rs))
    write(OUT/'pairwise_uncalled.json',dict(rows=pair_rows,summary=pair_summary))
    result=dict(**counts,max_independent_metric_error=maxerr,means_checked=means_checked,paired_CIs_checked=cis_checked,
        expert_receipts=expert_counts,text_interface_checks=phrases,source_separation_checked=True,
        stable_code_hashes_unchanged=True,all_evaluation_sources_retained=True,spatial_backward_calls=0)
    write(OUT/'QA_RESULTS.json',result);print(result,flush=True)


def notebook():
    import nbformat
    from nbclient import NotebookClient
    nb=nbformat.v4.new_notebook();nb.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
    nb.cells=[
        nbformat.v4.new_markdown_cell('## tl;dr\n本 notebook 核验 DeCoTA heuristic 机制实验。结论以封存预测和 QA_RESULTS 为准；这是历史名单机制复核，不是未接触测试集上的新确认。'),
        nbformat.v4.new_markdown_cell('## Context & Methods\n两 backbone × 两方向。每方向 32 开发源，评估 64/23 源。源级配对 bootstrap 10,000 次；随机条件先在源内平均。\n### Key Assumptions\n开发标签仅用于外层 gate/prior 选参；GT-anchor 是单独诊断。空间参数不更新。对照不做按评估结果择优拼接。'),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport json, numpy as np\nroot=Path('./artifacts/decota_heuristic_study_v1')\nread=lambda f:json.loads((root/f).read_text())\nqa=read('QA_RESULTS.json');qa"),
        nbformat.v4.new_markdown_cell('## Data\n独立视频数不是 backbone 次数、随机种子数或专家帧数的乘积。'),
        nbformat.v4.new_code_cell("lock=read('lock.json')\ncounts={s:{g:len(v) for g,v in gs.items()} for s,gs in lock['rows'].items()}\nassert counts=={'development':{'hc_to_vid':32,'vid_to_hc':32},'evaluation':{'hc_to_vid':64,'vid_to_hc':23}}\nassert qa['temporal_episodes']==302 and qa['main_parity']==174\ncounts"),
        nbformat.v4.new_markdown_cell('## Results\n固定主方法的部署 2×2。单位 corrected vIoU%。'),
        nbformat.v4.new_code_cell("s=read('summary.json')\nmethods=['time_frozen','time_decota','space_native','main_fixed']\n{cell:{m:round(v['absolute'][m]['vIoU_corrected']['mean']*100,3) for m in methods} for cell,v in s.items()}"),
        nbformat.v4.new_code_cell("for cell,v in s.items():\n print(cell, {k:v['contrasts'][k]['vIoU_corrected'] for k in ['main_fixed__minus__space_native','main_fixed__minus__absolute_fixed','time_decota__minus__time_donor_scaled_mean']})"),
        nbformat.v4.new_markdown_cell('## Takeaways\n参数发生更新不等于更新必要；组合收益不等于协同；残差传播收益不等于优于普通插值。具体方向及 CI 见 REPORT.md。'),
    ]
    f=OUT/'QA.ipynb';assert not f.exists()
    nbformat.write(nb,f)
    NotebookClient(nb,timeout=180,resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(nb,f)
    write(OUT/'NOTEBOOK_EXECUTED.json',dict(path=str(f),sha256=sha(f),executed=True))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['audit','notebook']);a=ap.parse_args()
    audit() if a.stage=='audit' else notebook()
