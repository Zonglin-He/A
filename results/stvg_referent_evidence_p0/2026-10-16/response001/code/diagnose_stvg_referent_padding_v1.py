"""One posthoc valid-content sensitivity; does not reopen the primary gate."""
from pathlib import Path
import collections,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_referent_evidence_p0_v1 import BASE,PRIOR,read,put,sha
def main():
    import numpy as np
    from scripts.stvg_referent_response_v1 import ranks,top
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    assert read(BASE/'VALID_CONTENT_SENSITIVITY_LOCK.json')['frozen_main_gate_unchanged']
    rr=read(BASE/'ROSTER_LOCK.json')['records'];cases={r['probe_alias']:load_case(r)[0]for r in rr};v=BASE/'valid_content';v.mkdir(exist_ok=False)
    for n in ('ROSTER_LOCK.json','INPUT_BARRIER.json'):(v/n).write_bytes((BASE/n).read_bytes())
    bar=read(BASE/'RESPONSE_GLOBAL_BARRIER.json');records=[];padding=[]
    for item in bar['records']:
        p=ROOT/item['path'];assert sha(p)==item['sha256'];a=np.load(p,allow_pickle=False);T,H,W=item['merged_grid'];c=cases[item['case_id']];ww,hh=c['input']['width'],c['input']['height'];sc=min(448/ww,448/hh);rw,rh=round(ww*sc),round(hh*sc);ox,oy=(448-rw)//2,(448-rh)//2
        y,x=np.meshgrid(np.arange(H),np.arange(W),indexing='ij');centers=np.stack([(x+.5)*28,(y+.5)*28],-1).reshape(-1,2);valid=(centers[:,0]>=ox)&(centers[:,0]<ox+rw)&(centers[:,1]>=oy)&(centers[:,1]<oy+rh);valid_ids=np.flatnonzero(valid);K=int(a['budget_per_cell']);assert len(valid_ids)>=K
        rq=np.full((T,H*W),-np.inf);rc=rq.copy()
        for t in range(T):rq[t,valid]=ranks(a['query_scores'][t,valid]);rc[t,valid]=ranks(a['caption_scores'][t,valid])
        consensus=np.minimum(rq,rc);disagreement=np.full_like(rq,-np.inf);disagreement[:,valid]=np.abs(rq[:,valid]-rc[:,valid]);scores={'query':rq,'caption':rc,'consensus':consensus,'disagreement':disagreement}
        # Swapped scores are re-derived from original projected/text vectors.
        z=a['features'];ss=a['swapped_embedding'];z=z/np.maximum(np.linalg.norm(z,axis=-1,keepdims=True),1e-6);ss=ss/np.maximum(np.linalg.norm(ss,axis=-1,keepdims=True),1e-6);m=z@ss.T;k=min(2,len(ss));raw=np.sort(m,axis=-1)[...,-k:].mean(-1);rs=np.full_like(rq,-np.inf)
        for t in range(T):rs[t,valid]=ranks(raw[t,valid])
        scores['query_swap']=rs;sel={name:np.stack([top(x,K)for x in score])for name,score in scores.items()}
        edges=np.array([e for e in a['edges']if valid[int(e[1])]and valid[int(e[2])]],dtype=float).reshape(-1,4);es=np.full((T,H*W),-np.inf);es[:,valid]=0
        for t,i,j,weight in edges:
            t,i,j=int(t),int(i),int(j);es[t,i]=max(es[t,i],weight);es[t+1,j]=max(es[t+1,j],weight)
        conn=[]
        for t in range(T):
            first=top(consensus[t],(K+1)//2);last=top(es[t],K-len(first),first);conn.append(np.r_[first,last])
        sel['connection']=np.stack(conn)
        for name,selected in sel.items():
            assert all(len(set(row))==K and valid[row].all()for row in selected)
            # Independent Python sorting verifies the rank/order implementation.
            for t,row in enumerate(selected):
                if name!='connection':expected=sorted(valid_ids,key=lambda i:(-float(scores[name][t,i]),int(i)))[:K]
                else:
                    first=sorted(valid_ids,key=lambda i:(-float(consensus[t,i]),int(i)))[:(K+1)//2];expected=first+sorted([i for i in valid_ids if i not in first],key=lambda i:(-float(es[t,i]),int(i)))[:K-len(first)]
                assert list(row)==expected
            original=a['selection_'+name];padding.append(dict(case_id=item['case_id'],candidate_id=item['candidate_id'],strategy=name,padding_center_fraction=float((~valid[original]).mean()),GT_read=False))
        p=v/'responses'/item['case_id']/(item['candidate_id']+'.npz');p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,edges=edges,cuts=a['cuts'],**{'selection_'+k:x for k,x in sel.items()})
        records.append(dict(item,path=str(p.relative_to(ROOT)),sha256=sha(p),original_response_sha256=item['sha256']));a.close()
    put(v/'RESPONSE_GLOBAL_BARRIER.json',dict(status='complete',count=16,records=records,GT_read=False,new_visual_encoder_passes=0,new_decoder_calls=0,new_student_updates=0,original_barrier_sha256=sha(BASE/'RESPONSE_GLOBAL_BARRIER.json')))
    put(v/'ROOT_RESPONSE_READBACK.json',dict(status='pass',count=16,clip_count=64,selections_valid_unique_same_budget=True,independent_python_sort_exact=True,GT_read=False))
    put(v/'GLOBAL_SELECTION_BARRIER.json',dict(status='complete',count=16,GT_read=False,root_sha256=sha(v/'ROOT_RESPONSE_READBACK.json')))
    put(BASE/'PADDING_POSTSEAL_DIAGNOSIS.json',dict(status='complete',count=16,records=padding,GT_read=False,scope='physical-image versus known marked-video letterbox padding; no target labels',sensitivity_lock_sha256=sha(BASE/'VALID_CONTENT_SENSITIVITY_LOCK.json')))
    from scripts import review_stvg_referent_evidence_p0_v1 as r
    r.BASE=v;r.geometry()
    put(v/'SENSITIVITY_COMPLETION.json',dict(status='complete',count=16,posthoc=True,GT_read=True,does_not_replace_primary_gate=True,main_decision_sha256=sha(BASE/'PHASE2_DECISION.json'),variant_result_sha256=sha(v/'RESPONSE_POSTSEAL_DIAGNOSIS.json'),scope='implementation-support sensitivity after fixed panel review, not blind confirmation or permission to rescue A/B/C'))
if __name__=='__main__':main()
