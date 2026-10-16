"""Fixed references four-arm backtracking comparison; no new Teacher."""
from pathlib import Path
import inspect,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_AB_backtrack002 import BASE,ABC,read,put,sha,check
def main():
    from scripts import audit_stvg_B_transfer_v1 as audit,score_stvg_B_transfer_v1 as score
    ns=dict(audit.__dict__);ns.update(BASE=BASE,PRIOR=ABC,check=check);src=inspect.getsource(audit.prefix).replace("'trials/B/STUDENT_GLOBAL_BARRIER.json'","'trials/A/STUDENT_GLOBAL_BARRIER.json'").replace("arm == 'R01'","arm == 'ABT'").replace("arm=='R01'","arm=='ABT'");exec(compile(src,str(Path(__file__)),'exec'),ns);ns['prefix']('ABT')
    ns=dict(score.__dict__);ns.update(BASE=BASE,PRIOR=ABC,check=check);src=inspect.getsource(score.main).replace("'trials/B'","'trials/A'").replace("'B' if name == 'B00'","'A' if name == 'B00'")
    # Source code is compact, so replace its actual ternary spelling as well.
    src=src.replace("'B'if name=='B00'","'A'if name=='B00'");exec(compile(src,str(Path(__file__)),'exec'),ns);ns['main']('ABT')
    d=read(BASE/'trials/ABT/POSTSEAL_RESULTS.json');bb=read(ROOT/'artifacts/stvg_B_transfer_v1/trials/R01/POSTSEAL_RESULTS.json');rows=[]
    for r in d['rows']:
        if r['arm']in ('Frozen','Temporal','B00','ABT'):
            x=dict(r);x['arm']={'B00':'A_Full','ABT':'A_BT'}.get(r['arm'],r['arm']);rows.append(x)
    for r in bb['rows']:
        if r['arm']in ('B00','R01'):
            x=dict(r);x['arm']={'B00':'B_Full','R01':'B_BT'}[r['arm']];rows.append(x)
    from scripts.score_stvg_referent_abc_v1 import aggregate
    s,p=aggregate(rows);idx={(r['case_id'],r['arm']):r for r in rows};effects=[]
    for ds in ('hc2','vidstg'):
        ids=sorted({r['case_id']for r in rows if r['dataset']==ds});v={cid:100*(idx[cid,'B_BT']['metrics']['m_vIoU']-idx[cid,'B_Full']['metrics']['m_vIoU']-idx[cid,'A_BT']['metrics']['m_vIoU']+idx[cid,'A_Full']['metrics']['m_vIoU'])for cid in ids};effects.append(dict(dataset=ds,contrast='(B_BT-B_Full)-(A_BT-A_Full)',mean_delta_pp=math.fsum(v.values())/8,per_source_delta_pp=v))
    put(BASE/'STUDY_READBACK.json',dict(status='pass',count=16,rows=rows,summary=s,paired=p,interactions=effects,GT_read=True,limits='same refs within A or B; A/B refs can differ; exact common-prefix before rescue audited; not semantic truth or new standalone samples'));print(json.dumps(dict(status='pass',interactions=effects)))
if __name__=='__main__':main()
