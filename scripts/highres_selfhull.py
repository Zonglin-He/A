"""Posthoc fixed cached control: a second view versus the hull operation itself.

No new encoder/decoder/head/expert. Fresh predictions seal before new labels.
"""
import os,sys,json,argparse,hashlib,csv
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.highres_priority.run import SUP,read,write,sha,validate as parent_validate
from stvg.temporal import mass_table,final_mbr
from scripts.tube_fusion.core import restore_geometry
from stvg.spatial import evidence_dp
OUT=ROOT/'outputs/highres_selfhull_v1';PUBLIC=ROOT/'results/highres_selfhull_v1'
ARMS=['SelfHull0','SelfHull1','SelfFusion0','SelfFusion1']

def validate():
    p=read(OUT/'plan.json');raw={k:v for k,v in p.items()if k!='fingerprint'};assert hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest()==p['fingerprint']
    assert sha(Path(__file__))==p['source_sha256'];assert sha(SUP/'plan.json')==p['parent_plan_sha256'] and sha(SUP/'sealed.json')==p['parent_seal_sha256'] and sha(ROOT/'results/highres_priority_v2/report.json')==p['primary_report_sha256'];parent_validate();return p

def prepare():
    assert not(OUT/'plan.json').exists();p=parent_validate();s=read(SUP/'sealed.json');assert s['queries']==63
    q=dict(schema='posthoc_same_offset_hull_v1',source_sha256=sha(Path(__file__)),parent_plan_sha256=sha(SUP/'plan.json'),parent_seal_sha256=sha(SUP/'sealed.json'),primary_report_sha256=sha(ROOT/'results/highres_priority_v2/report.json'),items=p['items'],labels=p['labels'],protocol='Posthoc mechanism control after main63 scores: same offset posterior drawn independently twice, original hull operator, original physical grid/MBR/BoxFixed. Four fixed arms offset0/1 hull with/without global224. Offset0 primary, offset1 sensitivity; no GT choice, new model, temperature/weight sweep or automatic adoption. All63 new cached decisions seal before follow-up scoring.')
    q['fingerprint']=hashlib.sha256(json.dumps(q,sort_keys=True).encode()).hexdigest();write(OUT/'plan.json',q);print('prepared',q['fingerprint'])

@torch.no_grad()
def predict():
    torch.set_num_threads(1);p=validate();assert not(OUT/'sealed.json').exists();entries=[]
    for item in p['items']:
        key=item['anonymous'];receipt=read(SUP/'receipts'/f'{key}.json');assert sha(SUP/receipt['posterior_path'])==receipt['posterior_sha256'];v=torch.load(SUP/receipt['posterior_path'],map_location='cpu',weights_only=False);cap=torch.load(ROOT/item['inputs']['capture'],map_location='cpu',weights_only=False);geom=restore_geometry(cap['geometry']);ids=v['frame_ids'];loc={int(x):i for i,x in enumerate(ids)};qs={}
        for offset in [0,1]:
            grid=v['grids'][offset];sub,small=mass_table([grid,grid]);q=np.zeros((len(ids),len(ids)),float);ix=[loc[int(x)]for x in sub];q[np.ix_(ix,ix)]=small;assert abs(q.sum()-1)<1e-12
            qs['SelfHull'+str(offset)]=q;qs['SelfFusion'+str(offset)]=.5*v['posteriors']['Only224six']+.5*q
        with np.load(SUP/receipt['prediction_path'])as old:values=dict(frame_ids=old['frame_ids'],positions8=old['positions8'],FixedSpatial=old['FixedSpatial'])
        for arm,q in qs.items():
            span=final_mbr(ids,q)['interval'];values[arm]=evidence_dp(geom.for_interval(span))['final'].numpy();values[arm+'_interval']=span
        pp=OUT/'predictions'/f'{key}.npz';pp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(pp,**values);entries.append(dict(query=key,path=str(pp.relative_to(OUT)),sha256=sha(pp)))
    assert len(entries)==63;validate();write(OUT/'sealed.json',dict(fingerprint=p['fingerprint'],files=entries,queries=63,GT_read=False,new_network_calls=0));print('sealed63 cached controls before labels')

def score():
    from scripts.interval_pool.score import score_boxes,summarize
    from scripts.joint_boundary_readout.score import paired
    torch.set_num_threads(1);p=validate();seal=read(OUT/'sealed.json');assert seal['fingerprint']==p['fingerprint'] and not seal['GT_read'] and seal['queries']==63
    assert sha(ROOT/p['labels']['path'])==p['labels']['sha256'];labels={r['id']:r for r in [json.loads(l)for l in (ROOT/p['labels']['path']).read_text().splitlines()if l.strip()]};base=read(ROOT/'results/highres_priority_v2/report.json');br={r['query']:r for r in base['queries_anonymous']};rows=[];signed=[];checks=0;error=0.
    for item,rec in zip(p['items'],seal['files']):
        assert sha(OUT/rec['path'])==rec['sha256'];key=rec['query'];gt=labels[item['id']];truth={int(k):v for k,v in gt['boxes'].items()};z=np.load(OUT/rec['path']);values=dict(br[key]['values']);fixed=dict(br[key]['fixed_spatial'])
        for arm in ARMS:
            span=z[arm+'_interval'].tolist();values[arm],e=score_boxes(z[arm],item['record'],truth,span,gt['interval'],z['positions8'].tolist());error=max(error,e);checks+=1;fixed[arm],e=score_boxes(z['FixedSpatial'],item['record'],truth,span,gt['interval'],z['positions8'].tolist());error=max(error,e);checks+=1
        rows.append(dict(query=key,parent=gt['parent'],values=values,fixed_spatial=fixed))
        for a,b in [('SelfHull0','SingleHigh0'),('SelfHull1','SingleHigh1'),('SelfFusion0','SingleFusion0'),('SelfFusion1','SingleFusion1'),('SelfFusion0','Fusion'),('SelfFusion1','Fusion')]:
            x,y=values[a],values[b];signed.append(dict(query=key,method=a,baseline=b,v_delta_pp=100*(x['v']-y['v']),t_delta_pp=100*(x['t']-y['t']),fixed_space_v_delta_pp=100*(fixed[a]['v']-fixed[b]['v']),length_delta_seconds=x['interval_length_seconds']-y['interval_length_seconds'],start_error_delta_seconds=x['start_error_seconds']-y['start_error_seconds'],end_error_delta_seconds=x['end_error_seconds']-y['end_error_seconds'],rescued_half=y['t']<.5<=x['t'],harmed_half=x['t']<.5<=y['t'],severe_v_loss=x['v']-y['v'] < -.05))
    assert len(rows)==len({r['parent']for r in rows})==63 and checks==504
    names=['Only224six','Only448','Fusion','SingleHigh0','SingleHigh1','SingleFusion0','SingleFusion1',*ARMS];pairs=[('SelfHull0','SingleHigh0'),('SelfHull1','SingleHigh1'),('SelfHull0','Only448'),('SelfFusion0','SingleFusion0'),('SelfFusion1','SingleFusion1'),('SelfFusion0','Fusion'),('SelfFusion1','Fusion'),('SelfFusion0','Only224six')]
    result=dict(status='completed',queries=63,exposure='Explicit posthoc mechanism control after primary63 scores; not preregistered main result or independent confirmation. No sweep, fixed offset0 primary.',means=summarize(rows,names,'values'),fixed_spatial_means=summarize(rows,names,'fixed_spatial'),paired={a+'-'+b:paired(rows,a,b)for a,b in pairs},queries_anonymous=[{k:v for k,v in r.items()if k!='parent'}for r in rows],proof=dict(all63_followup_predictions_sealed_before_new_labels=True,seal_sha256=sha(OUT/'sealed.json'),source_plan_fingerprint=p['fingerprint'],official_dense_checks=checks,official_dense_max_error=error,new_model_calls=0),production_changed=False)
    PUBLIC.mkdir(parents=True,exist_ok=True);write(PUBLIC/'report.json',result)
    with(PUBLIC/'signed_queries.tsv').open('w')as f:
        w=csv.DictWriter(f,fieldnames=list(signed[0]),delimiter='\t');w.writeheader();w.writerows(signed)
    lines=['# Same-offset hull: posthoc cached control','',result['exposure'],'','| Method | tIoU % | Full vIoU % | Mean interval s |','|---|---:|---:|---:|']
    for name,m in result['means'].items():lines.append(f"| {name} | {100*m['m_tIoU']:.4f} | {100*m['m_vIoU']:.4f} | {m['interval_length_seconds']:.4f} |")
    lines+=['','Four new cached arms, no new image/text/head/decoder/expert. Independent draws of the same view are enveloped by the ORIGINAL hull operator. This widens a posterior without adding second-view visual evidence. No per-query branch selection; all signed losses and fixed-space effects are retained.']
    (PUBLIC/'report.md').write_text('\n'.join(lines)+'\n');write(OUT/'score_receipt.json',dict(status='scored',report_sha256=sha(PUBLIC/'report.json')));print(json.dumps(dict(means=result['means'],paired=result['paired']),indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','predict','score']);x=a.parse_args();globals()[x.action]()
