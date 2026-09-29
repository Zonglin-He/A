"""Independent scalar-cosine reconstruction of causal O3 retrieval and writes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,math,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write,load,sha
from scripts.run_tastvg_preference_memory_o3_v1 import OLD,OUT,verify


def run():
    p=verify();rows=read(OUT/'ONLINE.json');assert sha(OUT/'ONLINE.json')==read(OUT/'ONLINE_BARRIER.json')['sha256'];memory=[];checks=0;maxerr=0.
    def digest():
        h=hashlib.sha256()
        for z in memory:h.update(np.asarray(z[0],dtype='<f8').tobytes());h.update(str(z[2]).encode())
        return h.hexdigest()
    for r in rows:
        pos=r['position'];x=load(OLD/'capture'/f'{pos:03}.pt');phi=x['phi'].numpy();base=x['base_score'].tolist();wins=[0]*len(phi)
        assert r['arrival_memory_size']==len(memory) and r['arrival_memory_hash']==digest()
        for z in r['pairs']:
            i,j=z['i'],z['j'];v=phi[i]-phi[j];n=float(np.linalg.norm(v));sims=[]
            if n and memory:
                direction=v/n
                sims=[math.fsum(float(a)*float(b) for a,b in zip(direction,m[0])) for m in memory]
                ids=sorted(range(len(memory)),key=lambda k:(-abs(sims[k]),k))[:3]
            else:ids=[]
            assert ids==[v['memory_index'] for v in z['neighbors']]
            val=math.fsum(sims[k]*memory[k][1] for k in ids);err=abs(val-z['vote']);maxerr=max(maxerr,err);assert err<1e-12
            y=1 if val>0 else (-1 if val<0 else 0);assert y==z['predicted_preference']
            if y:wins[i if y>0 else j]+=1
            for k,nb in zip(ids,z['neighbors']):assert memory[k][2][0]<pos and nb['source_position']==memory[k][2][0]
            checks+=1
        assert wins==r['arrival_wins'];winner=max(range(len(phi)),key=lambda k:(wins[k],base[k],-k));assert winner==r['arrival_selected']
        if r['expert']:
            e=load(OLD/'teacher/online'/f'{pos:03}.pt');assert r['selected']['Fast + Preference Memory']==r['selected']['Budgeted Rerank']==e['selected']
            for i in range(len(phi)):
                for j in range(i+1,len(phi)):
                    gap=e['scores'][i]-e['scores'][j];v=phi[i]-phi[j];n=float(np.linalg.norm(v))
                    if abs(gap)>1e-12 and n:label=1 if gap>0 else -1;memory.append((v/n,label,(pos,i,j,label)))
        else:assert r['selected']['Fast + Preference Memory']==winner and r['selected']['Budgeted Rerank']==0
        assert r['after_memory_size']==len(memory) and r['after_memory_hash']==digest()
    write(OUT/'STATE_AUDIT.json',dict(status='pass',arrivals=32,expert_writes=8,nonexpert_teacher_reads=0,pair_readouts=checks,final_stored_pairs=len(memory),max_scalar_vote_error=maxerr,all_neighbor_provenance_strictly_past=True,hash_chain_reconstructed=True,time=time.time()))
    print('AUDIT PASS',checks,len(memory),maxerr)

if __name__=='__main__':run()
