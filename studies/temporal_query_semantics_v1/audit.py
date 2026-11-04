"""Independent temporal arithmetic, cached-array and pair/video statistic audit."""
from common import *

def reference(F,q):
 F=np.asarray(F,dtype=np.float64);n=len(F);w=np.array([q[:i+1,i:].sum()for i in range(n)]);center=np.where(np.ptp(F,axis=0)==0,F[0],np.mean(F,axis=0));z=F-center;norm=np.sqrt(np.sum(z*z,axis=1));x=np.zeros_like(z);mask=norm>0;x[mask]=z[mask]/norm[mask,None]
 if w.sum()==0 or (1-w).sum()==0 or not np.any(x):L=np.zeros(n)
 else:
  ep=np.sum(x*w[:,None],axis=0)/w.sum();bp=np.sum(x*(1-w[:,None]),axis=0)/(1-w).sum();L=np.log(np.maximum((1+np.sum(x*ep,axis=1))/2,1e-12)/np.maximum((1+np.sum(x*bp,axis=1))/2,1e-12))
 pp=np.r_[0.,L.cumsum()];s=pp[1:][None,:]-pp[:-1][:,None];legal=q>0
 if np.all(s[legal]==0):post=q.copy()
 else:
  a=np.log(q[legal])+s[legal];mass=np.exp(a-a.max());post=np.zeros_like(q);post[legal]=mass/mass.sum()
 return post,L

def interval_iou(a,b):
 overlap=max(0.,min(a[1],b[1])-max(a[0],b[0]));den=(a[1]-a[0])+(b[1]-b[0])-overlap;return float(overlap/den)if den>0 else 0.

def main():
 validate();sys.addaudithook(old.deny_gpu_media);torch.set_num_threads(1);start=time.perf_counter();seal=read(D/'sealed.json');er=read(D/'evaluation_receipt.json');assert er['seal_sha256']==sha(D/'sealed.json')and er['report_sha256']==sha(P/'report.json')and er['rows_sha256']==sha(D/'private_pair_rows.json');truth=old.selected.labels('vidstg');pairrows=read(D/'private_pair_rows.json');by={r['pair']:r for r in pairrows};nc=0;mx=0.;canonical=read(R/'data/annotations/vidstg_tastvg_eval.json');inv=read(D/'inventory_v3.json');selection={x['a']['id']:x for x in inv['selected']}
 for pair in read(D/'cohort.json'):
  resources(start);a,pa=source_inputs(pair['a']);b,pb=source_inputs(pair['b']);stored=[]
  for key,x,y,proof in [(pair['a'],a,b,pa),(pair['b'],b,a,pb)]:
   e=seal['files'][key];assert e['input_SHA']==proof and sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];cold=derive(x,y)
   with np.load(D/e['path'])as z:
    assert set(z.files)==set(cold)and all(np.array_equal(v,z[k])for k,v in cold.items());z={k:z[k].copy()for k in z.files}
   stored.append(z);nc+=len(cold)
   for arm,F,q in [('SourceROI',x['Fs'],x['q0']),('CorrectedROI',x['F'],x['q0']),('CrossFeatures',y['F'],x['q0']),('CrossPrior',x['F'],y['q0']),('CrossBoth',y['F'],y['q0']),('CommonPrior',x['F'],(x['q0']+y['q0'])/2)]:
    post,L=reference(F,q);delta=max(float(np.max(np.abs(post-z[arm+'_q']))),float(np.max(np.abs(L-z[arm+'_L']))));mx=max(mx,delta);assert delta<2e-12,(key,arm,delta);assert np.array_equal(np.asarray(map_interval(x['ids'],post)),z[arm+'_interval'])
   assert np.array_equal(z['CorrectedROI_q'],z['text_metadata_control_q'])and np.array_equal(z['CorrectedROI_L'],z['text_metadata_control_L'])and np.array_equal(z['input_q0'],z['neutral_features_q'])
  sel=selection[pair['a']];aa,bb=sel['a'],sel['b'];assert aa['target']==bb['target']==aa['relation']['subject_tid']==bb['relation']['subject_tid']and aa['relation']['object_tid']==bb['relation']['object_tid']and aa['frame_ids']==bb['frame_ids']and aa['video_sha']==bb['video_sha']and aa['relation']['predicate']!=bb['relation']['predicate'];assert canonical[str(aa['index'])]['sentence']['target_id']==aa['target']and canonical[str(bb['index'])]['sentence']['target_id']==bb['target']
  za,zb=stored;ga,gb=truth[pair['a']]['interval'],truth[pair['b']]['interval'];r=by[pair['pair']]
  for arm in ARMS:
   ia,ib=za[arm+'_interval'],zb[arm+'_interval'];u,v=interval_iou(ia,ga),interval_iou(ib,gb);mean=50*(u+v);margin=50*(u+v-interval_iou(ia,gb)-interval_iou(ib,ga));assert abs(r[arm+'_t_percent']-mean)<1e-12 and abs(r[arm+'_assignment_margin_pp']-margin)<1e-12;assert r[arm+'_correct_assignment_percent']==100.*(margin>1e-10)and r[arm+'_both_events_t03_percent']==100.*(u>=.3 and v>=.3)
 report=read(P/'report.json')['summary'];n=len(pairrows);assert n==32;draws=np.random.default_rng(20261104).multinomial(n,[1/n]*n,size=10000);checks=0
 for k,v in report['statistics'].items():
  x=np.array([np.nan if r[k]is None else r[k]for r in pairrows]);mask=np.isfinite(x);den=draws[:,mask].sum(1);boot=np.divide(draws[:,mask]@x[mask],den,out=np.full(10000,np.nan),where=den>0);boot=boot[np.isfinite(boot)]
  if not mask.any():assert v['mean']is None and v['CI95']is None;continue
  assert abs(v['mean']-float(x[mask].mean()))<1e-12 and np.allclose(v['CI95'],np.quantile(boot,[.025,.975]),atol=1e-12,rtol=0)and v['pairs']==int(mask.sum());checks+=1
 rec=dict(status='independent_64query_allarrays_inputSHA_neutral_and_textcontract_exact_independent_prototype_posteriors_intervalIoU_identityactionpairs_and_videoCIs_audited',queries=64,pairs=32,independent_videos=32,all_array_checks=nc,independent_reference_max_error=mx,independent_statistic_checks=checks,report_sha256=sha(P/'report.json'),seal_sha256=sha(D/'sealed.json'),evaluation_receipt_sha256=sha(D/'evaluation_receipt.json'),all_current_own_evidence_intervals_exact=True,all_target_actor_object_track_inputs_exact=True,code_sha256=sha(Path(__file__)),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_Source=0,new_expert=0,new_GPU=0,new_model=0,new_optimizer=0,new_official_tube_scoring=0)
 atomic(D/'audit_receipt.json',rec);atomic(P/'audit_receipt.json',rec);print(json.dumps(rec),flush=True)
if __name__=='__main__':main()
