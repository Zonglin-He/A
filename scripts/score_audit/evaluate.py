"""Offline GT evaluation only after proposal AND held-feedback readouts are sealed."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.score_audit.run import lock,validate,TR,PUB,V0
from scripts.state_opd.run import read,sha
from scripts.state_opd.evaluate import score_boxes,mean_table,paired_stats,write_json
from scripts.propagation_v0.diagnose import _node_ious,distance_bins,rank_state,query_ranking,bootstrap_mean
NAMES=['Source','Projection','CrossAnchor']

def summary_replacements(queries,name,group='all'):
    a=np.array([[q['replacements'][name][group][k] for k in ['accepted','harm','severe5','delta_sum']]for q in queries],float)
    n,h,sev,s=a.sum(0);rng=np.random.default_rng(20261021);draw=rng.integers(0,len(a),(10000,len(a)));res=a[draw].sum(1)
    vals=np.divide(res[:,1],res[:,0],out=np.full(len(res),np.nan),where=res[:,0]>0);vals=vals[np.isfinite(vals)]
    return dict(fold_node_replacements=int(n),fold_node_harm=int(h),harm_fraction=float(h/n)if n else None,harm_fraction_CI95=np.quantile(vals,[.025,.975]).tolist()if len(vals)else None,fold_node_over5pp=int(sev),mean_delta_IoU_pp=float(100*s/n)if n else None,scope='repeated fold-nodes; CI resamples parents with all eight folds together')

def main():
    L=lock();PS=validate('proposals',L);FS=validate('feedback',L)
    lp=ROOT/'data/evaluation/development_labels.jsonl';lh=sha(lp)
    assert lh==read(ROOT/'outputs/layout_reward_dev/scope_hc2/metrics.json')['labels_sha256']
    records=[json.loads(s)for s in (ROOT/'data/evaluation/development.jsonl').read_text().splitlines()if s.strip()];records=[r for r in records if r['dataset']=='hc2'][:64]
    labels={g['id']:g for g in [json.loads(s)for s in lp.read_text().splitlines()if s.strip()]};rows=[];queries=[];checks=0;err=0.
    for i,rec in enumerate(records):
        k=rec['id'];g=labels[k];truth={int(k):v for k,v in g['boxes'].items()}
        P=torch.load(TR/'proposals'/PS['files'][k]['path'],map_location='cpu',weights_only=False);F=torch.load(TR/'feedback'/FS['files'][k]['path'],map_location='cpu',weights_only=False)
        assert F['proposal_sha256']==PS['files'][k]['sha256'] and len(P['folds'])==len(F['folds'])==8
        folds=[];replacement={n:{z:dict(accepted=0,harm=0,severe5=0,delta_sum=0.)for z in ['all','observed','near','middle','far']}for n in NAMES[1:]};ranks={n:[]for n in NAMES[1:]}
        bins=distance_bins(P['frame_ids'].numpy(),P['positions8']);obs=np.zeros(len(P['source']),bool);obs[P['positions8']]=True;bins['observed']=obs
        for b,f in zip(P['folds'],F['folds']):
            vals={};outs={'Source':P['source'],**f['outputs']}
            for name in NAMES:
                vals[name],e,_=score_boxes(np.asarray(outs[name]),rec,truth,P['interval'].tolist(),g['interval'],P['positions8']);checks+=1;err=max(err,e)
            folds.append(vals);ious=_node_ious(b['boxes'].numpy(),rec,truth)
            for name,s in [('Projection',b['projection']-.1*b['guard']),('CrossAnchor',f['cross']-.1*b['guard'])]:
                acts=f['actions'][name].numpy();known=np.isfinite(ious[:,0]);eligible=f['eligible'].numpy()
                d=ious[np.arange(len(acts)),acts]-ious[:,0]
                for group in replacement[name]:
                    mask=known & (acts!=0) & (np.ones(len(acts),bool)if group=='all'else bins[group])
                    rr=replacement[name][group];rr['accepted']+=int(mask.sum());rr['harm']+=int((d[mask]<-1e-12).sum());rr['severe5']+=int((d[mask]<-.05).sum());rr['delta_sum']+=float(d[mask].sum())
                state=[rank_state(np.asarray(s[t]),ious[t],b['valid'][t].numpy())for t in np.flatnonzero(known & eligible)]
                ranks[name].extend(state)
        # Statistical unit remains one parent, averaging the eight scores before any bootstrap.
        values={n:{field:None if not(v:=[x[n][field]for x in folds if x[n][field] is not None])else float(np.mean(v))for field in folds[0][n]}for n in NAMES}
        rows.append(dict(id=k,parent=g['parent'],values=values));queries.append(dict(query=f'H{i+1:03d}',values=values,replacements=replacement,ranking={n:query_ranking(ranks[n])for n in ranks},fold_values=folds))
    assert len(rows)==len({r['parent']for r in rows})==64
    means,counts=mean_table(rows,NAMES);pairs=paired_stats(rows,NAMES)
    proof=dict(fingerprint=L['fingerprint'],proposal_seal_sha256=sha(TR/'proposals/sealed.json'),feedback_seal_sha256=sha(TR/'feedback/sealed.json'),labels_sha256=lh,official_dense_checks=checks,max_error=err,proposals_sealed_before_held_feedback=True,readouts_sealed_before_GT=True,GT_read_for_prediction=False,updates=0,heavy_calls=0)
    public=dict(status='completed_development_scoring_probe_not_adapted_method',queries=64,folds=512,config=L['config'],means=means,valid_counts=counts,paired=pairs,replacements={n:{z:summary_replacements(queries,n,z)for z in ['all','observed','near','middle','far']}for n in NAMES[1:]},ranking={n:dict(macro_pair_accuracy=bootstrap_mean([q['ranking'][n]['pair_accuracy']for q in queries]),macro_Spearman=bootstrap_mean([q['ranking'][n]['Spearman']for q in queries]))for n in NAMES[1:]},queries_anonymous=queries,proof=proof)
    assert validate('proposals',L)==PS and validate('feedback',L)==FS and sha(lp)==lh and not torch.cuda.is_initialized()
    write_json(TR/'metrics.json',{**public,'private_rows':rows});write_json(PUB/'cross_anchor.json',public)
    lines=['# Strict proposal-holdout cross-anchor scoring probe','','All 64 exposed HC development parents × eight leave-one-observation-out folds are complete. No adaptation was executed. Each fold uses raw Source, seven-row query selection and seven teacher masks/boxes for construction. The held teacher mask enters only after every proposal and predicted support map is sealed; GT enters only after all scoring readouts are sealed. The held node itself always remains Source to exclude self-matching.','','This is cached teacher-output withholding, not independent seven-frame teacher inference: original ReferDINO outputs contain eight-frame tracking and temporal context. This is a scoring probe using seven observations for construction plus the eighth as feedback, not an unseen-teacher test or an OPD result. The new bank uses Source rather than ERef; compare its two matched scores to each other, not to old ERef-based v0 as a single-factor method gain. No memory or link term is used in either matched probe.','','| Score-only readout | vIoU % | Observed sIoU % | Unobserved sIoU % |','|---|---:|---:|---:|']
    for n in NAMES:lines.append('| '+n+' | '+' | '.join(f'{100*means[n][k]:.4f}'for k in ['m_vIoU','m_sIoU_observed_actual','m_sIoU_unobserved_actual'])+' |')
    lines+=['','| Compared with Projection | Delta pp | 95% parent interval pp |','|---|---:|---:|']
    for field in ['v','unobserved_actual_sIoU']:
        v=pairs['CrossAnchor-Projection'][field];lines.append(f'| {field} | {v["mean_pp"]:+.4f} | {v["CI95_pp"]} |')
    lines+=['','| Score | Accepted fold-node replacements with GT | Harm fraction | Mean replacement IoU delta pp |','|---|---:|---:|---:|']
    for n in NAMES[1:]:
        r=public['replacements'][n]['all'];lines.append(f'| {n} | {r["fold_node_replacements"]} | {100*r["harm_fraction"]:.3f}% | {r["mean_delta_IoU_pp"]:+.4f} |')
    lines+=['','Metrics are averaged within parent over all eight folds before parent bootstrap (10,000 draws). Repeated fold-nodes are explicitly counted as such; no claim of 512 independent videos is made. All positive/negative/unchanged anonymous query outcomes, fixed original-eight observation/unobserved partitions, near/middle/far bins, candidate ranking and fingerprints are in cross_anchor.json. GT never chooses a query-specific parameter, stopping step or output. Full-frame repeated-feature aliasing can still reward an oversized box; synthetic counterexamples are retained.']
    (PUB/'cross_anchor.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(means=means,paired=pairs['CrossAnchor-Projection'],replacements=public['replacements'])))
if __name__=='__main__':main()
