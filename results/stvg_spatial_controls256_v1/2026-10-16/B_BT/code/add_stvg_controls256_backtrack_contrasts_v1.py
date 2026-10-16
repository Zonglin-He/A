"""Publish saved-scalar B controls and the matched backtracking interaction."""
from pathlib import Path
import hashlib,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.score_stvg_spatial_controls256_v1 import validate_barrier
BASE=ROOT/'artifacts/stvg_spatial_controls256_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def saved(arm):
    validate_barrier(arm);b=BASE/'trials'/arm;p=b/'POSTSEAL_RESULTS002.json';c=json.loads((b/'CPU_COMPLETION002.json').read_text())
    assert c['status']=='complete' and c['count']==256 and c['GT_read'] and c['result_sha256']==sha(p)
    d=json.loads(p.read_text());return {(r['case_id'],r['arm']):r for r in d['rows']},sha(p)
def main(arm):
    assert arm in ('A_BT','B_BT')
    out=ROOT/'results/stvg_spatial_controls256_v1/2026-10-16'/arm
    roster=json.loads((BASE/'ROSTER_LOCK.json').read_text())['records'];aliases={r['probe_alias']:f'R{i:03d}' for i,r in enumerate(roster,1)}
    names=list(dict.fromkeys((arm,'A','B')))
    interaction=(BASE/'trials/B_BT/CPU_COMPLETION002.json').exists() and (BASE/'trials/A_BT/CPU_COMPLETION002.json').exists()
    if interaction:names=list(dict.fromkeys(names+['A_BT','B_BT']))
    data={};bindings={}
    for a in names:data[a],bindings[a]=saved(a)
    for cid in aliases:
        for control in ('Frozen','Temporal','Full_E1','Full_E1_Spatial'):
            first=data[arm][cid,control]
            for a in names:assert first['metrics']==data[a][cid,control]['metrics'] and first['dataset']==data[a][cid,control]['dataset']
    rows=[]
    for ds in ('hc2','vidstg'):
        ids=[r['probe_alias'] for r in roster if r['dataset']==ds];assert len(ids)==128
        metrics=tuple(data[arm][ids[0],arm]['metrics'])
        formulas=[(arm+'_minus_B',[(arm,1),('B',-1)])]
        if interaction:formulas.append(('backtracking_interaction_B_minus_A',[('B_BT',1),('B',-1),('A_BT',-1),('A',1)]))
        for label,terms in formulas:
            for metric in metrics:
                vals={}
                for cid in ids:
                    v=[(data[a][cid,a]['metrics'][metric],sign) for a,sign in terms]
                    if all(x is not None for x,_ in v):vals[aliases[cid]]=100*math.fsum(x*sign for x,sign in v)
                rows.append(dict(dataset=ds,comparison=label,metric=metric,valid=len(vals),unknown=128-len(vals),mean_delta_pp=math.fsum(vals.values())/len(vals) if vals else None,positive=sum(x>1e-6 for x in vals.values()),negative=sum(x< -1e-6 for x in vals.values()),neutral=sum(abs(x)<=1e-6 for x in vals.values()),per_source_delta_pp=vals))
    p=out/'BACKTRACK_MATCHED_SCALAR_READBACK.json';assert not p.exists()
    p.write_text(json.dumps(dict(status='pass',count=256,CPU_only=True,GT_read=False,new_GT_reader_calls=0,new_model_calls=0,shared_controls_all_metrics_exact=True,source_scores_sha256=bindings,matched_support='same prospective E1 positions; no cross-arm adopted-support merge',backtracking_interaction_available=interaction,paired=rows),indent=2)+'\n')
    doc=ROOT/'docs'/f'STVG_CORE256_{arm}_001.md';text=doc.read_text()+'\nAdditional matched B control and, when available, (B_BT - B) - (A_BT - A) use only already sealed scalar results. All shared controls are independently exact-checked. No new model or GT reader is called.\n\n| Domain | Contrast | vIoU delta (pp) | Positive / negative / unchanged / unknown |\n|---|---|---:|---|\n'
    for r in rows:
        if r['metric']=='m_vIoU':text+=f"| {r['dataset']} | {r['comparison']} | {r['mean_delta_pp']:+.6f} | {r['positive']} / {r['negative']} / {r['neutral']} / {r['unknown']} |\n"
    if not interaction:text+='\nThe B backtracking arm is still pending; this batch does not claim a completed 2 x 2 interaction.\n'
    doc.write_text(text)
    code=out/'code'/Path(__file__).name;code.write_bytes(Path(__file__).read_bytes())
    m=BASE/'publication'/arm/'EXPORT_MANIFEST.json';d=json.loads(m.read_text());paths=set(d['paths'])|{str(p.relative_to(ROOT)),str(code.relative_to(ROOT))};d.update(paths=sorted(paths),files=[dict(path=f,sha256=sha(ROOT/f),bytes=(ROOT/f).stat().st_size) for f in sorted(paths)]);m.write_text(json.dumps(d,indent=2)+'\n')
    print(json.dumps(dict(status='pass',count=256,backtracking_interaction_available=interaction,new_GT_reader_calls=0,new_model_calls=0)))
if __name__=='__main__':main(sys.argv[1])
