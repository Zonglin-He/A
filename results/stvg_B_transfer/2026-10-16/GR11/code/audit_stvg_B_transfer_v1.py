"""Saved B acceptance diagnosis, explicit scalar-only outputs; no model/GT."""
from pathlib import Path
import gc,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_B_transfer_v1 import BASE,PRIOR,read,put,sha,check
def baseline():
    import torch,numpy as np
    old=PRIOR/'trials/B';bar=read(old/'GLOBAL_PREDICTION_BARRIER.json');assert bar['status']=='complete'and bar['count']==16;rows=[]
    for item in bar['files']:
        p=ROOT/item['path'];assert sha(p)==item['sha256'];d=torch.load(p,map_location='cpu',weights_only=False);f=d['fits']['M2'];steps=[]
        if f is not None:
            for row in f['path']:
                before=row.get('reference_iou_before');before_mean=float(before.double().mean())if before is not None else None;cc=[]
                for c in row.get('candidates',[]):
                    after=float(c['anchor_IoU'].double().mean());cc.append(dict(scale=c['scale'],finite=c['finite'],reference_pass=c['reference_pass'],accepted=c['accepted'],reference_agreement_before=before_mean,reference_agreement_after=after,gain=after-before_mean))
                steps.append(dict(step=row['step'],accepted=row['accepted'],skipped=row['skipped'],reason=row.get('reason'),rollback=row['rollback'],accepted_scale=row['accepted_scale'],candidates=cc))
        rows.append(dict(case_id=d['case_id'],dataset=d['dataset'],actual_fit=f is not None,nonzero_final=any(not torch.equal(v.cpu(),f['final_state'][k].cpu())for k,v in f['initial_state'].items())if f else False,accepted_steps=f['optimizer_steps']if f else 0,steps=steps,GT_read=False));del d,f;gc.collect()
    put(BASE/'BASELINE_ACCEPTANCE_READBACK.json',dict(status='pass',count=16,records=rows,prior_barrier_sha256=sha(old/'GLOBAL_PREDICTION_BARRIER.json'),auditor_sha256=sha(Path(__file__)),GT_read=False,scope='all B saved proposal agreement/finite/rollback and early-stop scalars; no raw states or new decoding'))
    print(json.dumps(dict(status='pass',count=16,actual_fits=sum(r['actual_fit']for r in rows),nonzero=sum(r['nonzero_final']for r in rows),rejected_queries=sum(any(not c['accepted']for s in r['steps']for c in s['candidates'])for r in rows))))
def prefix(arm):
    import torch,numpy as np
    check(arm);b=BASE/'trials'/arm;assert read(b/'ROOT_READBACK.json')['status']=='pass';old={x['case_id']:x for x in read(PRIOR/'trials/B/STUDENT_GLOBAL_BARRIER.json')['files']};rows=[]
    for item in read(b/'STUDENT_GLOBAL_BARRIER.json')['files']:
        x=torch.load(ROOT/item['path'],map_location='cpu',weights_only=False);y=torch.load(ROOT/old[item['case_id']]['path'],map_location='cpu',weights_only=False);a=x['fits']['M2'];o=y['fits']['M2'];seen=[];diverged=False
        if arm=='R01'and a is not None and o is not None:
            for n,(u,v)in enumerate(zip(a['path'],o['path'])):
                if diverged or u['skipped']or v['skipped']:break
                err={}
                for key in ('boxes_before','gradient','original_adam_delta','filtered_delta'):
                    if u.get(key)is None or v.get(key)is None:assert u.get(key)is v.get(key);continue
                    err[key]=float((u[key].double()-v[key].double()).abs().max());assert err[key]<=2e-6,(item['case_id'],n,key,err[key])
                for key in ('samples','rewards','weights'):
                    e=float((u['rollout'][key].double()-v['rollout'][key].double()).abs().max());assert e<=3e-6;err['rollout_'+key]=e
                assert u['candidates'][0]['accepted']==v['candidates'][0]['accepted'];first_failed=not v['candidates'][0]['accepted'];rescued=first_failed and u['accepted'];seen.append(dict(step=n,max_abs_error=err,original_full_rejected=first_failed,same_first_full_acceptance=True,smaller_scale_rescued=rescued,accepted_scale=u['accepted_scale']));diverged=rescued
        rows.append(dict(case_id=x['case_id'],dataset=x['dataset'],same_reference_common_prefix=seen,GT_read=False));del x,y,a,o;gc.collect()
    put(b/'PREFIX_READBACK.json',dict(status='pass',count=16,records=rows,GT_read=False,scope='R01 only: original full-proposal common-prefix arithmetic before backtracking trajectory divergence; not an independent backward'))
    print('PREFIX_READBACK_PASS',arm)
if __name__=='__main__':baseline()if sys.argv[1]=='baseline'else prefix(sys.argv[2])
