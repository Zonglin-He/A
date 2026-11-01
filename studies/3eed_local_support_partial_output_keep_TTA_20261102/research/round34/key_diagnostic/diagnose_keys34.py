"""CPU-only, GT-free discrimination / precision audit of source retrieval keys.
No predictions, training, model states, or held-out U/FINAL inputs are accessed.
"""
import os
for v in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[v]='4'
import sys,time,json,hashlib,collections
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(4)
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'crossplatform/round32'))
from common32 import rows,feature
BASE=Path(__file__).resolve().parent
KINDS=['q64_noun_interaction','q64_projection_only','h288_preprojection']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):Path(p).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def norm(a):return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)
def keys(z,kind,dtype):
    q=np.asarray(z['query'] if kind!='h288_preprojection' else z['h'],dtype=dtype)
    if kind=='q64_noun_interaction':
        t=np.asarray(z['tokens'],dtype=dtype)[z['token_mask']].mean(0);q=q*norm(t)
    return norm(q)
def stats(a):
    a=np.asarray(a,dtype=np.float64).ravel();assert np.isfinite(a).all()
    return {'n':int(a.size),'min':float(a.min()),'p01':float(np.quantile(a,.01)),'p10':float(np.quantile(a,.1)),'median':float(np.median(a)),'p90':float(np.quantile(a,.9)),'p99':float(np.quantile(a,.99)),'max':float(a.max()),'mean':float(a.mean()),'fractions_le':{str(t):float((a<=t).mean()) for t in [0,1e-8,1e-7,5e-7,1e-6,1e-5,1e-4,.001,.01]}}
def neighbor(Q,K,exclude=None):
    sim=(Q@K.reshape(-1,K.shape[-1]).T).reshape(len(Q),len(K),K.shape[1]);which=sim.argmax(-1);near=np.take_along_axis(sim,which[:,:,None],axis=-1)[:,:,0]
    if exclude is not None:near[:,exclude]=-np.inf
    obs=np.argsort(-near,axis=-1,kind='stable')[:,:5];best=np.take_along_axis(near,obs,axis=-1);pos=np.take_along_axis(which,obs,axis=-1)
    top=sim[np.arange(len(Q)),obs[:,0]];top2=np.partition(top,-2,axis=-1)[:,-2:];qgap=top2[:,-1]-top2[:,-2]
    return {'obs':obs,'pos':pos,'best':best,'row_gap1':best[:,0]-best[:,1],'row_gap4':best[:,3]-best[:,4],'query_gap':qgap}
def main():
    started=time.perf_counter();out=BASE/'attempt001';out.mkdir(exist_ok=False)
    rr=[dict(r,diagnostic_split=s) for s in ['A208','DEV64','T128'] for r in rows(s)];assert len(rr)==400 and len({r['id'] for r in rr})==400
    assert not any(r['stage'] in ['U','F','S','R'] for r in rr)
    feedback_path=ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json';fb=json.loads(feedback_path.read_text());assert len(fb)==22
    spec={'state':'locked before diagnostic output','splits':{'A208':208,'DEV64':64,'T128':128},'key_kinds':KINDS,'precisions':['numpy_float32','numpy_float64'],'all256queries_per_observation':True,'A_self_row_excluded':True,'source_features_only':True,'query_scores_used_only_for_q0':True,'no_GT_no_U_no_FINAL':True,'new_model_or_prediction_states':0,'CPU_only':True,'threads':4,'feedback_sha256':sha(feedback_path),'program_sha256':sha(__file__),'thresholds_descriptive_only':True,'h288_key':'L2-normalized input to image contrastive projection MLP; point-text-conditioned decoder hidden activation','q64_noun_key':'L2-normalized elementwise product of source query projection and normalized source noun token mean','q64_only_key':'L2-normalized source query projection without noun product'}
    dump(out/'SPEC_LOCK.json',spec)
    zs={r['id']:feature(r) for r in rr};records=[];within=[];cross=[];private=[]
    for pl in ['drone','quad']:
        A=[r for r in rr if r['diagnostic_split']=='A208' and r['platform']==pl];lookup={r['id']:i for i,r in enumerate(A)};flag=np.zeros((len(A),256),bool)
        for f in fb:
            if f['platform']==pl:flag[lookup[f['id_PRIVATE']],f['positive_group_PRIVATE']+f['negative_group_PRIVATE']]=True
        for kind in KINDS:
            K32=np.stack([keys(zs[r['id']],kind,np.float32) for r in A]);K64=np.stack([keys(zs[r['id']],kind,np.float64) for r in A])
            # Fixed nonaligned query sample from distinct A observations, not a label-based subset.
            jjs=np.arange(0,256,32);QQ=K64[:,jjs].reshape(-1,K64.shape[-1]);sim=QQ@QQ.T;ob=np.repeat(np.arange(len(A)),len(jjs));v=sim[(ob[:,None]!=ob[None,:])&np.triu(np.ones(sim.shape,bool),1)]
            cross.append({'platform':pl,'key':kind,'sample_rule':'indices0,32,...,224 in every A row; only distinct rows','cosine':stats(v),'fractions_cos_gt':{str(t):float((v>t).mean()) for t in [.9,.99,.999,.9999,.999999]}})
            for r in [r for r in rr if r['platform']==pl]:
                z=zs[r['id']];Q32=keys(z,kind,np.float32);Q64=keys(z,kind,np.float64);q0=int(z['scores'].argmax());exclude=lookup.get(r['id']);n32=neighbor(Q32,K32,exclude);n64=neighbor(Q64,K64,exclude)
                G=Q64@Q64.T;off=G[np.triu_indices(256,1)];center=Q64-Q64.mean(0);C=center@center.T;pr=float(np.trace(C)**2/max(np.sum(C*C),1e-30));ng=G.copy();np.fill_diagonal(ng,-np.inf);nearest=ng.max(-1)
                within.append({'alias':r['alias'],'platform':pl,'split':r['diagnostic_split'],'key':kind,'offdiag_cosine':stats(off),'fraction_cos_gt':{str(t):float((off>t).mean()) for t in [.9,.99,.999,.9999,.999999]},'nearest_cosine':stats(nearest),'centered_participation_rank':pr,'exact_unique_key_rows':int(len(np.unique(Q64,axis=0)))})
                diffrow=n32['obs'][:,0]!=n64['obs'][:,0];diffendpoint=(n32['obs'][:,0]!=n64['obs'][:,0])|(n32['pos'][:,0]!=n64['pos'][:,0]);gate32=flag[n32['obs'],n32['pos']];gate64=flag[n64['obs'],n64['pos']]
                rec={'alias':r['alias'],'platform':pl,'split':r['diagnostic_split'],'key':kind,'q0':q0,'all_query_row_margin':stats(n64['row_gap1']),'all_query_row_k4_boundary_margin':stats(n64['row_gap4']),'all_query_within_chosen_row_candidate_margin':stats(n64['query_gap']),'all_query_nearest_A_cosine':stats(n64['best'][:,0]),'CPU32_vs_CPU64_row_changes':int(diffrow.sum()),'CPU32_vs_CPU64_row_or_candidate_changes':int(diffendpoint.sum()),'CPU32_vs_CPU64_k1_gate_changes':int((gate32[:,0]!=gate64[:,0]).sum()),'CPU32_vs_CPU64_k4_gate_changes':int((gate32[:,:4].any(-1)!=gate64[:,:4].any(-1)).sum()),'q0_row_changed':bool(diffrow[q0]),'q0_endpoint_changed':bool(diffendpoint[q0]),'q0_gate_changed':bool(gate32[q0,0]!=gate64[q0,0]),'q0_row_margin64':float(n64['row_gap1'][q0]),'q0_candidate_margin64':float(n64['query_gap'][q0]),'k1_gate_fraction64':float(gate64[:,0].mean()),'k4_gate_fraction64':float(gate64[:,:4].any(-1).mean())};records.append(rec)
                private.append({'alias':r['alias'],'key':kind,'neighbors32_Arowindex':n32['obs'].tolist(),'candidate32':n32['pos'].tolist(),'neighbors64_Arowindex':n64['obs'].tolist(),'candidate64':n64['pos'].tolist()})
            print('KEY34',pl,kind,'seconds',round(time.perf_counter()-started,1),flush=True)
    dump(out/'OBSERVATION_ROWS.json',records);dump(out/'WITHIN_CANDIDATE_ROWS.json',within);dump(out/'CROSS_OBSERVATION_SAMPLE.json',cross);dump(out/'NEIGHBORS_PRIVATE.json',private)
    summary=[]
    for pl in ['drone','quad']:
        for split in ['A208','DEV64','T128']:
            for kind in KINDS:
                a=[r for r in records if (r['platform'],r['split'],r['key'])==(pl,split,kind)];w=[r for r in within if (r['platform'],r['split'],r['key'])==(pl,split,kind)]
                summary.append({'platform':pl,'split':split,'key':kind,'observations':len(a),'queries':len(a)*256,'mean_within_pair_cosine':float(np.mean([r['offdiag_cosine']['mean'] for r in w])),'median_within_pair_cosine_rowmedians':float(np.median([r['offdiag_cosine']['median'] for r in w])),'median_centered_participation_rank':float(np.median([r['centered_participation_rank'] for r in w])),'mean_fraction_within_pairs_cos_gt_09999':float(np.mean([r['fraction_cos_gt']['0.9999'] for r in w])),'median_nearest_A_row_margin_rowmedians':float(np.median([r['all_query_row_margin']['median'] for r in a])),'median_chosen_row_candidate_margin_rowmedians':float(np.median([r['all_query_within_chosen_row_candidate_margin']['median'] for r in a])),'CPU32_vs_CPU64_row_changes':sum(r['CPU32_vs_CPU64_row_changes'] for r in a),'CPU32_vs_CPU64_row_or_candidate_changes':sum(r['CPU32_vs_CPU64_row_or_candidate_changes'] for r in a),'CPU32_vs_CPU64_k1_gate_changes':sum(r['CPU32_vs_CPU64_k1_gate_changes'] for r in a),'CPU32_vs_CPU64_k4_gate_changes':sum(r['CPU32_vs_CPU64_k4_gate_changes'] for r in a),'q0_row_changes':sum(r['q0_row_changed'] for r in a),'q0_endpoint_changes':sum(r['q0_endpoint_changed'] for r in a),'q0_gate_changes':sum(r['q0_gate_changed'] for r in a),'mean_k1_gate_fraction64':float(np.mean([r['k1_gate_fraction64'] for r in a]))})
    dump(out/'SUMMARY.json',summary);manifest={p.name:sha(p) for p in out.iterdir() if p.is_file()};dump(out/'SEALED.json',{'state':'GT_FREE_CPU_DIAGNOSTIC_COMPLETED','sha256':manifest,'seconds':time.perf_counter()-started,'n_observations':400,'n_candidate_key_comparisons':400*256*3,'no_GPU_no_GT_no_U_no_FINAL':True});print('KEY34_SEALED',flush=True)
if __name__=='__main__':main()
