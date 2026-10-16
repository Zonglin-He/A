"""Original postseal Student metrics plus exact A and Full prompt controls."""
from pathlib import Path
import copy,inspect,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_spatial_who_order_v1 import BASE,ABC,PRIOR,read,put,sha,verify
def lock():
    # Lock the scorer before opening any new ground truth or final scores.
    p=BASE/'SCORER_LOCK.json';put(p,dict(status='locked',scorer_sha256=sha(Path(__file__)),prior_scorer_sha256=sha(ROOT/'scripts/score_stvg_referent_abc_v1.py'),GT_read=False))
def main(arm):
    assert sha(Path(__file__))==read(BASE/'SCORER_LOCK.json')['scorer_sha256'];verify()
    from scripts import score_stvg_referent_abc_v1 as old
    # Namespace-only adaptation of original metric and locked semantic reader.
    ns=dict(old.__dict__);ns.update(BASE=BASE,verify=verify)
    exec(compile(inspect.getsource(old.semantic),str(Path(__file__)),'exec'),ns);exec(compile(inspect.getsource(old.main),str(Path(__file__)),'exec'),ns);ns['main'](arm)
    b=BASE/'trials'/arm;d=read(b/'POSTSEAL_RESULTS.json');p=ABC/'trials/A';bar=read(p/'GLOBAL_PREDICTION_BARRIER.json');assert read(p/'ROOT_READBACK.json')['status']=='pass'and bar['root_sha256']==sha(p/'ROOT_READBACK.json')
    for x in bar['files']:assert sha(ROOT/x['path'])==x['sha256']and sha(ROOT/x['receipt_path'])==x['receipt_sha256']
    a=read(p/'POSTSEAL_RESULTS.json');assert read(p/'CPU_COMPLETION.json')['result_sha256']==sha(p/'POSTSEAL_RESULTS.json');d['rows']+=[copy.deepcopy(r)for r in a['rows']if r['arm']in ('A','A_Spatial')];d['summary'],d['paired']=old.aggregate(d['rows']);d['semantic']['rows']+=[r for r in a['semantic']['rows']if r['arm']=='A'];d['semantic']['summary']+=[r for r in a['semantic']['summary']if r['arm']=='A']
    d.update(prompt_scorer_sha256=sha(Path(__file__)),paired_original_control='Full_E1'if arm=='FULL_ORDER'else'A',control_A_predictions_exact=True,GT_read=True)
    # Geometry remains scalar-only at exact known positions. Source candidate is1.
    for r in d['geometry']:
        vals={x['candidate_id']:x['GT_iou']for x in r['candidates']};s=r['selections'][arm];v=s.get('selected_candidate_id');src=vals.get(1);sel=vals.get(v);r['selected_minus_Source_IoU_pp']=100*(sel-src)if sel is not None and src is not None else None
    put(b/'POSTSEAL_RESULTS_EXTENDED.json',d);put(b/'EXTENDED_CPU_COMPLETION.json',dict(status='complete',count=16,row_count=len(d['rows']),result_sha256=sha(b/'POSTSEAL_RESULTS_EXTENDED.json'),GT_read=True));print(json.dumps(dict(status='complete',arm=arm,paired=[r for r in d['paired']if r['metric']=='m_vIoU'and r['comparison']in(arm+'_minus_'+d['paired_original_control'],arm+'_minus_Temporal')],semantic_summary=d['semantic']['summary'])))
if __name__=='__main__':lock()if sys.argv[1]=='lock'else main(sys.argv[1])
