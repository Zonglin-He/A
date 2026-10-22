"""Post-confirmation matched-candidate ranking; diagnostic, never selection feedback."""
import os,sys,time,json
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from research.target_semantic128_shrink_v1.run import R,D,P,A,read,put,sha,PROVIDER
from research.target_semantic128_shrink_v1.score import truths

def pairs_and_quality(q0,ids,truth):
 ij=np.stack(np.nonzero(q0>0),1);v=q0[ij[:,0],ij[:,1]];take=np.argsort(-v,kind='stable')[:64];ij=ij[take];v=v[take];ids=np.asarray(ids);lo=ids[ij[:,0]];hi=ids[ij[:,1]]+1;inter=np.maximum(0,np.minimum(hi,truth[1])-np.maximum(lo,truth[0]));quality=inter/(np.maximum(hi,truth[1])-np.minimum(lo,truth[0]));ii,jj=np.triu_indices(len(ij),1);good=(ij[ii,1]-ij[ii,0]==ij[jj,1]-ij[jj,0])&(np.maximum(v[ii],v[jj])<=2*np.minimum(v[ii],v[jj]))&(abs(quality[ii]-quality[jj])>1e-12)
 return ij,quality,ii[good],jj[good],v

def main():
 torch.set_num_threads(1);start=time.perf_counter();f=read(D/'frozen_config.json');receipt=read(D/'confirm_receipt.json');assert receipt['report_sha256']==sha(P/'confirm_report.json');s=read(D/'sealed.json');p=read(D/'plan.json');source=read(A/'sealed.json');rows=[]
 for ds in ['hc2','vidstg']:
  items=[x for x in p['items']if x['dataset']==ds];lf=p['labels'][ds];assert sha(PROVIDER/lf['path'])==lf['sha256'];gt=truths(PROVIDER/lf['path'],{x['row']['id']for x in items})
  for item in items:
   key=item['alias'];e=s['files'][key];ae=source['files'][key];assert sha(D/e['trace_path'])==e['trace_sha256']and sha(A/ae['path'])==ae['sha256'];tr=torch.load(D/e['trace_path'],map_location='cpu',weights_only=False);old=torch.load(A/ae['path'],map_location='cpu',weights_only=False);cap=torch.load(A/ae['capture_path'],map_location='cpu',weights_only=False);q0=tr['q']['NativeG6'];ij,quality,i,j,v=pairs_and_quality(q0,cap['frame_ids'],gt[item['row']['id']]['interval']);scores={};actual={};beta=tr['steps'][f['weak']]['beta']
   for a in ['Loop1','Loop2',f['shrink'],f['weak']]:
    step=tr['steps'][a][-1]if a=='Loop2'else tr['steps'][a];S=step['S']*(beta if a==f['weak']else 1.);scores[a]=S[ij[:,0],ij[:,1]];actual[a]=np.log(tr['q'][a][ij[:,0],ij[:,1]].clip(1e-300))
   actual['NativeG6']=np.log(v);direction=np.sign(quality[i]-quality[j]);accuracy={}
   for mode,values in [('evidence_only',scores),('posterior',actual)]:
    accuracy[mode]={a:float(np.mean((1+direction*np.sign(z[i]-z[j]))/2))if len(i)else None for a,z in values.items()}
   duration=np.asarray(cap['frame_ids'])[ij[:,1]]+1-np.asarray(cap['frame_ids'])[ij[:,0]];rows.append(dict(query=key,dataset=ds,split=item['split'],matched_pairs=len(i),same_grid_span_not_exact_physical_duration_max_gap=int(np.max(abs(duration[i]-duration[j])))if len(i)else None,accuracy=accuracy,oracle_top64_t=float(quality.max()),native_selected_t=next(x for x in read(P/(item['split']+'_report.json'))['domains'][ds]['rows']if x['query']==key)['values']['NativeG6']['t']))
 summary={}
 for split in ['develop','confirm']:
  for ds in ['hc2','vidstg']:
   rr=[x for x in rows if x['split']==split and x['dataset']==ds];valid=[x for x in rr if x['matched_pairs']]
   if not valid:
    summary[split+'_'+ds]=dict(queries=64,valid_queries=0,pairs=0,means=None,shrink_minus_one=None);continue
   ix=np.random.default_rng(20261023).integers(0,len(valid),(10000,len(valid)));summary[split+'_'+ds]=dict(queries=64,valid_queries=len(valid),excluded_no_quality_disagreement=64-len(valid),pairs=sum(x['matched_pairs']for x in rr),means={mode:{a:float(np.mean([x['accuracy'][mode][a]for x in valid]))for a in valid[0]['accuracy'][mode]}for mode in ['evidence_only','posterior']},shrink_minus_one={mode:dict(mean=float(np.mean(delta:=np.array([x['accuracy'][mode][f['shrink']]-x['accuracy'][mode]['Loop1']for x in valid]))),CI95=np.quantile(delta[ix].mean(1),[.025,.975]).tolist())for mode in ['evidence_only','posterior']})
 out=dict(kind='Post-confirmationGTdiagnostic_NOTselection/onlineoracle',code_sha256=sha(Path(__file__)),frozen_config_sha256=sha(D/'frozen_config.json'),confirm_receipt_sha256=sha(D/'confirm_receipt.json'),candidate_rule='Top64 originalq0 cells; equalgridspan and priorratio<=2; all nonzero truequality differences. No GT candidates selected for inference.',raw_weak_scaling_cannot_add_rank_information=True,summary=summary,rows=rows,new_model=0,new_predictions=0,seconds=time.perf_counter()-start);put(P/'CANDIDATE_RANK_AUDIT.json',out);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
