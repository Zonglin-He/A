"""Prelocked actual retained endpoint/path accounting; no model or GT."""
from pathlib import Path
import collections,hashlib,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_referent_abc002 import BASE as ABC,read,put,sha,verify
BASE=ROOT/'artifacts/stvg_BC_retained_edges_v1'
def lock():
    BASE.mkdir(exist_ok=False);jobs=read(ABC/'TEACHER_JOB_LOCK.json')['records'];pins={}
    for j in jobs:
        p=ABC/'native_cache'/j['job_id']/'POLICIES.json';pins[str(p.relative_to(ROOT))]=sha(p)
    put(BASE/'DESIGN_LOCK.json',dict(status='locked',count=16,window_count=64,GT_read=False,arms=['B','C'],edge='saved positive reciprocal same-candidate local projected-feature edge; not semantic identity',edge_measure='sum weights with both actual retained endpoints / all positive saved edge weights; no graph => unknown',path_measure='two consecutive edges through same middle vertex, min(weight1,weight2); all three actual retained vertices; context two cells not applicable',aggregation='window graph ratios mean within parent then equal eight parents per domain; context and local separately; no graph unknown, not zero',conditional_paired_C_trigger='one paired-endpoint construction only if local source-macro retained-edge ratio C minus B is <=0 in at least one domain with eight evaluable sources; otherwise not triggered; no both-domain benefit gate',no_bootstrap=True,no_new_GT=True,no_model=True))
    files=[Path(__file__),BASE/'DESIGN_LOCK.json',ABC/'TEACHER_JOB_LOCK.json',ABC/'teacher/B/ROOT_READBACK.json',ABC/'teacher/C/ROOT_READBACK.json'];pins.update({str(p.relative_to(ROOT)):sha(p)for p in files});put(BASE/'RUNTIME.json',dict(status='locked',pins=pins,GT_read=False));print('EDGE_PROTOCOL_LOCKED')
def main():
    for p,h in read(BASE/'RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h,p
    verify();rows=[]
    for j in read(ABC/'TEACHER_JOB_LOCK.json')['records']:
        p=ABC/'native_cache'/j['job_id']/'POLICIES.json';s=read(p)
        for arm in ('B','C'):assert read(ABC/'teacher'/arm/(j['job_id']+'.json'))['native_policy_sha256']==sha(p)
        for i,r in enumerate(s['records']):
            edges=[(int(t)*256+int(u),(int(t)+1)*256+int(v),float(w))for t,u,v,w,g in r['edges']if float(w)>0];allw=math.fsum(w for u,v,w in edges);paths=[]
            if i==1:
                left=collections.defaultdict(list)
                for u,v,w in edges:left[u].append((v,w))
                paths=[(u,v,z,min(w,q))for u,v,w in edges for z,q in left[v]]
            pathw=math.fsum(w for u,v,z,w in paths)
            for a in ('B','C'):
                keep=set(r['indices'][a]);kept=math.fsum(w for u,v,w in edges if u in keep and v in keep);pk=math.fsum(w for u,v,z,w in paths if u in keep and v in keep and z in keep)
                rows.append(dict(case_id=j['case_id'],dataset=j['dataset'],slot=j['center_slot'],view='context'if i==0 else'local',arm=a,edge_count=len(edges),retained_edge_count=sum(u in keep and v in keep for u,v,w in edges),total_edge_weight=allw,retained_edge_weight=kept,retained_edge_ratio=kept/allw if allw>0 else None,dangling_edge_count=sum((u in keep)!=(v in keep)for u,v,w in edges),two_edge_path_count=len(paths)if i==1 else None,retained_two_edge_path_count=sum(u in keep and v in keep and z in keep for u,v,z,w in paths)if i==1 else None,retained_path_weight_ratio=pk/pathw if pathw>0 else None,GT_read=False))
    sources=[];summary=[];dif=[]
    for ds in ('hc2','vidstg'):
        for view in ('context','local'):
            for arm in ('B','C'):
                ids=sorted({r['case_id']for r in rows if r['dataset']==ds});v=[]
                for cid in ids:
                    rr=[x for x in rows if x['case_id']==cid and x['view']==view and x['arm']==arm];known=[x['retained_edge_ratio']for x in rr if x['retained_edge_ratio']is not None];pv=[x['retained_path_weight_ratio']for x in rr if x['retained_path_weight_ratio']is not None];m=math.fsum(known)/len(known)if known else None;sources.append(dict(case_id=cid,dataset=ds,view=view,arm=arm,known_windows=len(known),unknown_windows=4-len(known),retained_edge_ratio=m,retained_path_weight_ratio=math.fsum(pv)/len(pv)if pv else None));v+=[m]if m is not None else[]
                summary.append(dict(dataset=ds,view=view,arm=arm,known_sources=len(v),unknown_sources=8-len(v),mean_retained_edge_ratio=math.fsum(v)/len(v)if v else None))
            b,c=[next(x for x in summary if x['dataset']==ds and x['view']==view and x['arm']==arm)for arm in ('B','C')];dif.append(dict(dataset=ds,view=view,known_sources=min(b['known_sources'],c['known_sources']),C_minus_B=c['mean_retained_edge_ratio']-b['mean_retained_edge_ratio']if b['mean_retained_edge_ratio']is not None and c['mean_retained_edge_ratio']is not None else None))
    eligible=all(x['known_sources']==8 for x in dif if x['view']=='local');trigger=eligible and any(x['C_minus_B']<=0 for x in dif if x['view']=='local');put(BASE/'CPU_COMPLETION.json',dict(status='complete',count=16,window_count=64,rows=rows,source_rows=sources,summary=summary,paired=dif,paired_C_triggered=trigger,trigger_evaluable=eligible,GT_read=False,CPU_only=True,limitations='feature correspondence is not semantic identity; preserving edges can preserve wrong/background structure; ratios do not establish Student benefit'))
    print(json.dumps(dict(status='complete',paired=dif,paired_C_triggered=trigger,GT_read=False)))
if __name__=='__main__':lock()if sys.argv[1]=='lock'else main()
