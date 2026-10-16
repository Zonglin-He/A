"""Matched factorial main effects and interaction from already scored groups."""
from pathlib import Path
import collections,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_B_transfer_v1 import BASE,read,put,sha
from scripts.score_stvg_referent_abc_v1 import aggregate
def main():
    rows={};actual={};usage=[];geometry={};bindings={}
    for a in ('G10','R01','GR11'):
        b=BASE/'trials'/a;d=read(b/'POSTSEAL_RESULTS.json');assert read(b/'CPU_COMPLETION.json')['result_sha256']==sha(b/'POSTSEAL_RESULTS.json');s,p=aggregate(d['rows']);assert s==d['summary']and p==d['paired'];assert read(b/'PREFIX_READBACK.json')['status']=='pass';bindings[a]=dict(result_sha256=sha(b/'POSTSEAL_RESULTS.json'),Student_root_sha256=sha(b/'ROOT_READBACK.json'),prefix_sha256=sha(b/'PREFIX_READBACK.json'));geometry[a]=d['geometry']
        for target,source in [(rows,d['rows']),(actual,d['actual_support_rows'])]:
            for r in source:
                k=r['case_id'],r['arm']
                if k in target:assert target[k]==r
                target[k]=r
        for r in d['usage']:
            if r['arm']==a:usage.append(r)
    assert len(rows)==len(actual)==160;assert geometry['G10']==geometry['GR11'];rows=list(rows.values());actual=list(actual.values());summary,paired=aggregate(rows);sa,pa=aggregate(actual);idx={(r['case_id'],r['arm']):r['metrics']for r in rows};sub=read(BASE/'SUBGROUP_LOCK.json');interactions=[];strata=[]
    for ds in ('hc2','vidstg'):
        ids=sorted({r['case_id']for r in rows if r['dataset']==ds})
        for metric in idx[ids[0],'B00']:
            vals={cid:100*(idx[cid,'GR11'][metric]-idx[cid,'G10'][metric]-idx[cid,'R01'][metric]+idx[cid,'B00'][metric])for cid in ids if all(idx[cid,a][metric]is not None for a in ('B00','G10','R01','GR11'))};v=list(vals.values());interactions.append(dict(dataset=ds,metric=metric,contrast='GR11-G10-R01+B00',valid=len(v),unknown=8-len(v),mean_delta_pp=math.fsum(v)/len(v)if v else None,per_source_delta_pp=vals))
        for group in ('original_fullstep_rejected','original_updates_accepted'):
            selected=[cid for cid in ids if sub['groups'][cid]==group]
            for a in ('G10','R01','GR11'):
                vv={cid:100*(idx[cid,a]['m_vIoU']-idx[cid,'B00']['m_vIoU'])for cid in selected};strata.append(dict(dataset=ds,group=group,arm=a,count=len(vv),mean_delta_pp=math.fsum(vv.values())/len(vv)if vv else None,per_source_delta_pp=vv))
    put(BASE/'STUDY_READBACK.json',dict(status='pass',count=16,rows=rows,summary=summary,paired=paired,actual_support_rows=actual,actual_support_summary=sa,actual_support_paired=pa,interactions=interactions,prelocked_subgroups=strata,usage=usage,geometry=geometry,bindings=bindings,GT_read=True,no_new_GT_or_model_execution=True,all_source_negative_unknown_retained=True))
    print(json.dumps(dict(status='pass',count=16,paired=[r for r in paired if r['metric']=='m_vIoU'and r['comparison']in ('G10_minus_B00','R01_minus_B00','GR11_minus_B00','GR11_minus_G10','GR11_minus_R01')],interactions=[r for r in interactions if r['metric']=='m_vIoU'])))
if __name__=='__main__':main()
