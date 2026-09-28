"""Full current-method descriptive analysis. Does not tune or change predictions."""
import argparse,collections,hashlib,json,math,sys,textwrap,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.stats import spearmanr
from scripts.decota_matrix_common_v1 import read,write,status,load,sha
from scripts.run_stvg_fullscale_v1 import plan,label_payload,OUT as SOURCE
from scripts.score_stvg_fullscale_v1 import score,summarize
from scripts.audit_decota_heuristic_study_v1 import independent_quality
from vg_tta.decota_full_case_diagnostics_v1 import data_features,temporal_contributions,spatial_contributions

OUT=ROOT/'artifacts/decota_full_case_iteration_v1'
COMPONENTS={'temporal':('temporal_only','frozen'),'spatial':('mymethod','temporal_only'),'joint':('mymethod','frozen')}
EPS=.001


def source_distinct(rows,key,reverse=False,limit=5):
    seen=set();result=[]
    for r in sorted(rows,key=key,reverse=reverse):
        if r['source'] in seen:continue
        result.append(r);seen.add(r['source'])
        if len(result)==limit:break
    return result


def pairing(rows,component):
    good=[r for r in rows if r['deltas'][component]>EPS];bad=[r for r in rows if r['deltas'][component]<-EPS]
    fields=['duration_seconds','gt_box_area_mean','gt_speed_target_units_per_second','caption_words','gt_center_fraction']
    if component=='spatial':fields.append('gt_event_fraction')
    matrix=np.array([[r['features'].get(k) if r['features'].get(k) is not None else np.nan for k in fields] for r in rows])
    median=np.nanmedian(matrix,0);scale=np.nanpercentile(matrix,75,axis=0)-np.nanpercentile(matrix,25,axis=0);scale=np.maximum(scale,1e-6)
    mapping={r['key']:np.nan_to_num((v-median)/scale,nan=0).clip(-10,10) for r,v in zip(rows,matrix)}
    pairs=[]
    for a in source_distinct(bad,key=lambda r:r['deltas'][component],limit=6):
        eligible=[r for r in good if r['source']!=a['source'] and r['query_type']==a['query_type']]
        if not eligible:continue
        b=min(eligible,key=lambda r:float(np.linalg.norm(mapping[r['key']]-mapping[a['key']])))
        pairs.append(dict(component=component,failure=a['key'],good=b['key'],matched_fields=fields,
            scaled_distance=float(np.linalg.norm(mapping[b['key']]-mapping[a['key']])),
            note='Retrospective descriptive nearest match, not causal identification; match quality may be poor'))
    return pairs


def associations(rows,component):
    fields=[k for k,v in rows[0]['features'].items() if v is None or isinstance(v,(int,float))]
    result=[]
    for f in fields:
        groups=collections.defaultdict(list)
        for r in rows:
            value=r['features'].get(f)
            if value is not None and np.isfinite(value) and r['deltas'][component] is not None:
                groups[r['source']].append((value,r['deltas'][component]))
        points=np.array([np.mean(v,axis=0) for _,v in sorted(groups.items())])
        if len(points)<8 or np.ptp(points[:,0])<1e-10 or np.ptp(points[:,1])<1e-10:continue
        rho=float(spearmanr(points[:,0],points[:,1]).statistic)
        good=[r for r in rows if r['deltas'][component] is not None and r['deltas'][component]>EPS and r['features'].get(f) is not None]
        bad=[r for r in rows if r['deltas'][component] is not None and r['deltas'][component]<-EPS and r['features'].get(f) is not None]
        # Source-equal means within each outcome group, not treating queries as independent.
        def groupmean(rs):
            z=collections.defaultdict(list)
            for r in rs:z[r['source']].append(r['features'][f])
            return float(np.mean([np.mean(v) for v in z.values()])) if z else None
        result.append(dict(feature=f,source_spearman=rho,eligible_sources=len(points),
            good_source_equal_mean=groupmean(good),failure_source_equal_mean=groupmean(bad),
            status='association_only',outcome_grouping_is_posthoc=True))
    return sorted(result,key=lambda z:abs(z['source_spearman']),reverse=True)


def report(dest,summary,ranked,pairs,cases):
    lines=['# TA-STVG 当前方法全量案例分析', '',
        '状态：'+('全量可用输入完成' if summary['all_available_complete'] else '仅冒烟快照，不是全量结论'),'',
        f"官方清单 {summary['nominal_queries']} 条；已评分 {summary['scored_queries']} 条，{summary['scored_sources']} 个源；不可用输入 {len(summary['unavailable'])} 条。",'',
        '原始测试数据在本轮被用于机制诊断；后续改法不能再把它称为未接触确认集。参数仍为原先锁定值。','',
        '| 范围 | 方法 | 查询均值 vIoU | 源等权 vIoU | sIoU 源等权 | tIoU 源等权 |','|---|---|---:|---:|---:|---:|']
    for scope,z in summary['groups'].items():
        for method,m in z['methods'].items():
            fmt=lambda x:'NA' if x is None else f'{100*x:.3f}'
            lines.append('| '+' | '.join([scope,method,fmt(m['vIoU_corrected']['query_mean']),fmt(m['vIoU_corrected']['mean']),fmt(m['sIoU']['mean']),fmt(m['tIoU']['mean'])])+' |')
    lines+=['','## 相对影响，不与绝对正确性混淆','',
        '| 组件 | 查询改善/中性/受损 | 源等权变化 pp | 95% CI pp |','|---|---|---:|---|']
    for c,z in summary['components'].items():
        ci=z['delta']['ci95'];ci='NA' if ci is None else str([round(100*x,3) for x in ci])
        lines.append(f"| {c} | {z['query_win_neutral_loss']} | {100*z['delta']['mean']:.3f} | {ci} |")
    lines+=['','## 表面关联：供排列竞争解释，不当作因果结论','']
    for c,items in ranked.items():
        lines += ['### '+c,'','| 数据/过程因素 | 源级 Spearman | good 组源等权均值 | failure 组源等权均值 |','|---|---:|---:|---:|']
        for z in items[:12]:
            fmt=lambda v:'NA' if v is None else f'{v:.4g}'
            lines.append(f"| {z['feature']} | {z['source_spearman']:.3f} | {fmt(z['good_source_equal_mean'])} | {fmt(z['failure_source_equal_mean'])} |")
    lines+=['','## 实际案例入口','',
        '案例包括各组件源隔离的强正/负样本、匹配对及固定seed随机参照；不是只挑符合假设的图片。',
        'frames 目录为真实输入网格帧。绿色GT，黄色Frozen框，红色完整方法框；时间选择状态单独标注。',
        '这些图片由程序生成，只有后续实际人工/模型查看后才能标为遮挡、切镜或身份错误。','',
        '| key | source | 选择原因 | temporal Δpp | spatial Δpp | joint Δpp |','|---|---|---|---:|---:|---:|']
    for c in cases:
        lines.append('| '+' | '.join([c['key'],c['source'],', '.join(c['selection_reasons']),*[f"{100*c['deltas'][k]:.3f}" for k in COMPONENTS]])+' |')
    lines+=['','## 数值审计与结论边界','',
        f"独立IoU/指标最大误差：{summary['independent_metric_max_error']}；时间/空间精确贡献分解最大误差：{summary['contribution_max_error']}。",
        'GT只进入本分析；未进行GT监督适应或按GT逐样本选择预测。原参数和生产代码未改。',
        '相关性排序不证明技术原因。候选的预测、反证条件和单因素验证计划见上级 HYPOTHESES.md；本报告不自动宣布算法失效或推荐部署。','']
    (dest/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')


def render_cases(dest,cases,bykey,max_cases=32):
    from PIL import Image,ImageDraw,ImageFont
    from vg_tta.exact_frame_decode_audit_v2 import decode
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    receipts=[]
    for case in cases[:max_cases]:
        name=case['key'].replace(':','_')+'.jpg';out=dest/'frames'/name
        if out.exists():
            receipts.append(dict(key=case['key'],path=str(out),sha256=sha(out),visually_reviewed=False));continue
        x=load(case['prediction_path']);q=x['input'];frames,ids=decode(q);gt=bykey[case['key']]
        pos=np.unique(np.rint(np.linspace(0,len(ids)-1,min(8,len(ids)))).astype(int));w,h=320,240
        canvas=Image.new('RGB',(4*w,130+2*(h+28)),(245,245,245));draw=ImageDraw.Draw(canvas)
        title=f"{case['key']} source={case['source']} | {'; '.join(case['selection_reasons'])}"
        draw.text((10,5),title,fill='black',font=font)
        for j,line in enumerate(textwrap.wrap(q['caption'],width=130)[:4]):draw.text((10,28+21*j),line,fill='black',font=font)
        for k,i in enumerate(pos):
            raw=Image.fromarray(frames[i]);raw.thumbnail((w,h));panel=Image.new('RGB',(w,h),(20,20,20));ox=(w-raw.width)//2;oy=(h-raw.height)//2;panel.paste(raw,(ox,oy));pd=ImageDraw.Draw(panel)
            def box(z,color):
                cx,cy,bw,bh=map(float,z);corners=[ox+(cx-bw/2)*raw.width,oy+(cy-bh/2)*raw.height,ox+(cx+bw/2)*raw.width,oy+(cy+bh/2)*raw.height]
                pd.rectangle(corners,outline=color,width=2)
            if gt['valid'][i]:box(gt['boxes'][i],(20,245,40))
            box(x['predictions']['frozen']['boxes'][i],(255,210,0));box(x['predictions']['mymethod']['boxes'][i],(255,35,35))
            px=(k%4)*w;py=130+(k//4)*(h+28);canvas.paste(panel,(px,py));ni=x['predictions']['frozen']['indices'];ai=x['predictions']['mymethod']['indices']
            draw.text((px+3,py+h+3),f"frame {ids[i]}  in time: F={ni[0]<=i<=ni[1]} A={ai[0]<=i<=ai[1]}",fill='black',font=font)
        out.parent.mkdir(parents=True,exist_ok=True);canvas.save(out,quality=90)
        receipts.append(dict(key=case['key'],path=str(out),sha256=sha(out),visually_reviewed=False))
    write(dest/'render_receipts.json',dict(renders=receipts,visually_reviewed=False))


def run(cohort,smoke=False,no_render=False):
    torch.set_num_threads(2);p=plan();source=SOURCE/'method/tastvg'/cohort
    if not smoke and not (source/'barrier.json').exists():raise RuntimeError('Full prediction barrier required before full analysis')
    dest=OUT/('smoke' if smoke else 'analysis')/cohort
    if (dest/'complete.json').exists():return
    if (dest/'summary.json').exists():
        if not no_render and not (dest/'render_receipts.json').exists():
            render_cases(dest,read(dest/'case_catalog.json'),label_payload(p),max_cases=4 if smoke else 40)
        write(dest/'complete.json',dict(summary_sha256=sha(dest/'summary.json'),rows_sha256=sha(dest/'rows.json'),
            cases_visually_reviewed=False,single_factor_experiments_started=False,recovered_postprocessing=True,created=time.time()))
        return
    labels=label_payload(p);rows=[];unavailable=[];maxerr=0.;maxcontrib=0.
    for path in sorted(source.glob('[0-9]*.json')):
        rr=read(path);assert sha(rr['path'])==rr['sha256'] and rr['lock_sha256']==sha(SOURCE/'lock.json')
        x=load(rr['path'])
        if x.get('status')=='input_unavailable':unavailable.append(dict(key=x['key'],reason=x['input_unavailable']));continue
        assert not x['GT_used'];gt=labels[x['key']];ids=x['frame_ids'];metrics={};qualities={}
        truth=np.asarray(gt['boxes']);valid=np.asarray(gt['valid'],bool)
        for name,z in x['predictions'].items():
            metrics[name],qualities[name]=score(z['boxes'],gt,ids,z['indices'])
            iq=independent_quality(z['boxes'],truth,valid)
            if valid.any():
                err=float(np.max(np.abs(iq[valid]-qualities[name][valid])));maxerr=max(maxerr,err);assert err<1e-9
        assert metrics['frozen']['sIoU']==metrics['temporal_only']['sIoU']
        assert metrics['temporal_only']['tIoU']==metrics['mymethod']['tIoU']
        deltas={c:metrics[a]['vIoU_corrected']-metrics[b]['vIoU_corrected'] if metrics[a]['vIoU_corrected'] is not None and metrics[b]['vIoU_corrected'] is not None else None for c,(a,b) in COMPONENTS.items()}
        tc=temporal_contributions(qualities['frozen'],valid,ids,gt['interval'],x['predictions']['frozen']['indices'],x['predictions']['temporal_only']['indices'])
        sc=spatial_contributions(qualities['frozen'],qualities['mymethod'],valid,ids,gt['interval'],x['predictions']['mymethod']['indices'],[z['position'] for z in x['pseudo']])
        for c,z in [('temporal',tc),('spatial',sc)]:
            if deltas[c] is not None:err=abs(deltas[c]-z['total']);maxcontrib=max(maxcontrib,err);assert err<1e-9
        feature=data_features(x,gt,qualities['frozen'],qualities['mymethod'])
        feature.update(native_span=metrics['frozen']['span'],adapted_span=metrics['mymethod']['span'],
            span_change=metrics['mymethod']['span']-metrics['frozen']['span'])
        rows.append(dict(key=x['key'],source=x['input']['source'],query_type=x['query_type'],caption=x['input']['caption'],
            prediction_path=rr['path'],prediction_sha256=rr['sha256'],metrics=metrics,deltas=deltas,features=feature,
            temporal_accounting=tc,spatial_accounting=sc,config=x['config'],seconds=x['seconds'],
            diagnostic_fold=int(hashlib.sha256(x['input']['source'].encode()).hexdigest()[:12],16)%5))
    assert rows
    if not smoke:assert len(rows)+len(unavailable)==len(p['rows'][cohort])
    eligible=[r for r in rows if all(d is not None for d in r['deltas'].values())]
    groups={}
    for tag,rs in [('all',rows),('caption',[r for r in rows if r['query_type']=='caption']),('question',[r for r in rows if r['query_type']=='question'])]:
        if not rs:continue
        ss=[r['source'] for r in rs]
        groups[tag]=dict(queries=len(rs),sources=len(set(ss)),methods={m:{k:summarize([r['metrics'][m][k] for r in rs],ss) for k in ['vIoU_corrected','sIoU','tIoU']} for m in ['frozen','temporal_only','mymethod']})
    component={}
    for c in COMPONENTS:
        vals=np.array([r['deltas'][c] for r in eligible]);ss=[r['source'] for r in eligible]
        component[c]=dict(delta=summarize(vals,ss),query_win_neutral_loss=[int((vals>EPS).sum()),int((abs(vals)<=EPS).sum()),int((vals<-EPS).sum())])
    ranked={c:associations(eligible,c) for c in COMPONENTS};pairs=[z for c in COMPONENTS for z in pairing(eligible,c)]
    selected=collections.defaultdict(list);bykey={r['key']:r for r in rows}
    for c in COMPONENTS:
        for label,rr in [('good',source_distinct([r for r in eligible if r['deltas'][c]>EPS],key=lambda r:r['deltas'][c],reverse=True)),
                         ('failure',source_distinct([r for r in eligible if r['deltas'][c]<-EPS],key=lambda r:r['deltas'][c]))]:
            for r in rr:selected[r['key']].append(c+'_'+label)
    for tag,reverse in [('absolute_best',True),('absolute_worst',False)]:
        for r in source_distinct(eligible,key=lambda r:r['metrics']['mymethod']['vIoU_corrected'],reverse=reverse,limit=5):selected[r['key']].append(tag)
    for r in source_distinct(eligible,key=lambda r:hashlib.sha256(('20260911:'+r['key']).encode()).hexdigest(),limit=8):selected[r['key']].append('random_reference')
    for z in pairs:
        selected[z['good']].append('matched_good');selected[z['failure']].append('matched_failure')
    cases=[{**bykey[k],'selection_reasons':sorted(set(v))} for k,v in selected.items()]
    cases.sort(key=lambda r:r['key'])
    summary=dict(cohort=cohort,backbone='tastvg',nominal_queries=len(p['rows'][cohort]),scored_queries=len(rows),
        scored_sources=len({r['source'] for r in rows}),unavailable=unavailable,all_available_complete=not smoke,
        full_nominal_complete=not smoke and not unavailable,groups=groups,components=component,
        independent_metric_max_error=maxerr,contribution_max_error=maxcontrib,GT_used_only_for_analysis=True,
        joint_metric_unavailable_queries=len(rows)-len(eligible),
        final_confirmation_claim=False,production_changed=False,created=time.time(),analysis_script_sha256=sha(Path(__file__)))
    write(dest/'rows.json',rows);write(dest/'summary.json',summary);write(dest/'ranked_associations.json',ranked)
    write(dest/'matched_pairs.json',pairs);write(dest/'case_catalog.json',cases)
    report(dest,summary,ranked,pairs,cases)
    if not no_render:render_cases(dest,cases,labels,max_cases=4 if smoke else 40)
    write(dest/'complete.json',dict(summary_sha256=sha(dest/'summary.json'),rows_sha256=sha(dest/'rows.json'),
        cases_visually_reviewed=False,single_factor_experiments_started=False,created=time.time()))
    print('FULL_CASE_ANALYSIS',cohort,'queries',len(rows),'sources',summary['scored_sources'],'smoke',smoke,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cohort',choices=['hcstvg1_test','vidstg_test'],required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--no-render',action='store_true');a=ap.parse_args();run(a.cohort,a.smoke,a.no_render)
