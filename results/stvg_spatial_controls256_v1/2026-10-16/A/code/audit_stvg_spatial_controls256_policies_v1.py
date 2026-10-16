"""Independent CPU replay of actual native selections and safe graph scalars."""
from pathlib import Path
import hashlib,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_stvg_spatial_controls256_v1 import BASE,read,put,sha
def main():
    import numpy as np
    from scripts.stvg_referent_abc_tokens002 import select
    from scripts.stvg_referent_abc_native002 import geometry
    jobs=read(BASE/'TEACHER_JOB_LOCK.json')['records'];dims=read(BASE/'SOURCE_GEOMETRY.json')['records'];rows=[]
    for j in jobs:
        b=BASE/'native_cache'/j['job_id'];s=read(b/'POLICIES.json');assert s['job_sha256']==hashlib.sha256(json.dumps(j,sort_keys=True).encode()).hexdigest()and sha(b/'native.npz')==s['native_sha256'];v,ma,gr=geometry(j,dims[j['case_id']]['width'],dims[j['case_id']]['height'])
        with np.load(b/'native.npz',allow_pickle=False)as a:
            for i,(start,stop)in enumerate(((0,512),(512,1280))):
                assert np.array_equal(v[i],a[f'valid{i}'])and np.array_equal(ma[i],a[f'mandatory{i}'])and np.array_equal(gr[i],a[f'groups{i}']);fresh=select(a['features'][start:stop].reshape(len(v[i]),256,-1),a['query_embedding'],a['caption_embedding'],v[i],ma[i],gr[i]);old=s['records'][i];assert fresh['quotas']==old['quotas']and fresh['role_quotas']==old['role_quotas']and fresh['edges']==old['edges']
                for k in ('A','B','C'):assert fresh['indices'][k].tolist()==old['indices'][k]
        rows.append(dict(case_id=j['case_id'],dataset=j['dataset'],slot=j['center_slot'],caption_available=s['caption_available'],all_three_indices_edges_roles_exact=True,GT_read=False))
    put(BASE/'POLICY_GLOBAL_ROOT.json',dict(status='pass',count=256,window_count=len(jobs),records=rows,all_source_geometry_and_policies_replayed_exact=True,GT_read=False,CPU_only=True));print('NATIVE256_POLICY_REPLAY_PASS',len(jobs))
if __name__=='__main__':main()
