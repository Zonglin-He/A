"""Independent full-grid audit, whole-domain labels and complete signed export."""
from common import *
from compatibility import current_exact
import argparse,csv
def arrays(e):
 path=Path(e['path']);path=path if path.is_absolute()else D/path;assert sha(path)==e['sha256']
 with np.load(path)as z:return {k:z[k].copy()for k in ['Matching','Current','PureMatched','Matching_interval','Current_interval','PureMatched_interval','frame_ids','positions8']}
def checked(ds,row,e,native=False):
 assert sha(D/e['receipt_path'])==e['receipt_sha256'];a=arrays(e);control,ch=controls(ds,row['id']);assert ch==e['control_sha256']and all(np.array_equal(a[k],x)for k,x in control.items());assert np.array_equal(a['Matching_interval'],a['Current_interval'])
 cp=Path(e['compact_path']);cp=cp if cp.is_absolute()else D/cp;assert sha(cp)==e['compact_sha256']
 if e['origin']['kind']=='original128_exact_reuse':
  old=fixed(PILOT/ds/'sealed.json')['files'][row['id']];assert e['origin']['pilot_receipt_sha256']==old['receipt_sha256']and e['sha256']==old['sha256']and e['compact_sha256']==old['compact_sha256'];assert fixed(PILOT/ds/'reporting_receipt.json')['queries']==128;return dict(original128_complete_audit_adopted=True,GT_read=False)
 v=torch.load(cp,map_location='cpu',weights_only=False,mmap=True);b,diag=replay(v,ds);assert np.array_equal(b,a['Matching'])and diag==e['diagnostics'];cap,par=load_parent(ds,row);assert par==e['parent'];ref=compact(cap,v['features'],v['valid'],v['shape'],row)
 for k in ['features','valid','seeds']:assert torch.equal(torch.as_tensor(v[k]),torch.as_tensor(ref[k]))
 assert np.array_equal(a['Matching'][v['positions8']],v['native_boxes'])
 if native:
  source=D/e['origin']['native_path'];assert sha(source)==e['origin']['native_sha256'];raw=torch.load(source,map_location='cpu',weights_only=False,mmap=True);f,valid,shape,proof=current_exact(cap,raw,control);assert torch.equal(f,v['features'])and torch.equal(valid,v['valid'])and tuple(shape)==tuple(v['shape'])and proof==e['origin']['original_native_and_current_exact'];bb,dd=replay(compact(cap,f,valid,shape,row),ds);assert np.array_equal(bb,a['Matching'])and dd==diag
 return dict(fullgrid_cold_exact=True,raw_native_Source_and_current_rederived=native,GT_read=False)
def bridge(ds):
 plan=check();torch.set_num_threads(1);sys.addaudithook(__import__('stvg.runtime',fromlist=['guard']).guard);dest=D/ds;j=read(dest/'first2_Source_input.json');assert len(j['keys'])==2 and not(dest/'actual_first2_Source_bridge.json').exists();by={r['id']:r for r in rows(ds)}
 cases=[checked(ds,by[key],j['files'][key],True)for key in j['keys']];atomic(dest/'actual_first2_Source_bridge.json',dict(status='passed_actual_two_new_original_Source_nativeheads_grid_current_Epool_geometry_interval_finalDP_VideoWalk_cold_exact',cases=cases,plan_fingerprint=plan['fingerprint'],worker=identity(),new_Source=0,new_expert=0,new_GT=0));print(ds,'actual newSource first2 bridge passed',flush=True)
def shard_audit(ds,shard):
 plan=check();torch.set_num_threads(1);sys.addaudithook(__import__('stvg.runtime',fromlist=['guard']).guard);dest=D/ds;file=dest/'shards'/f'{shard:05d}.json';j=read(file);by={r['id']:r for r in rows(ds)};proof=[]
 for key in j['keys']:
  e=j['files'][key];proof.append(checked(ds,by[key],e,e['origin']['kind']=='new_Source'))
 atomic(file.with_name(f'{shard:05d}_audit.json'),dict(status='independent_fullgrid_allarrays_originalSource_current_compatibility_exact_before_GT',queries=len(proof),cases=proof,all_arrays_cold_exact=True,shard_sha256=sha(file),worker=identity(),GT_read=False,plan_fingerprint=plan['fingerprint']));print(ds,shard,'independent fullgrid audit passed',flush=True)
def whole_input_audit(ds):
 check();torch.set_num_threads(1);sys.addaudithook(__import__('stvg.runtime',fromlist=['guard']).guard);dest=D/ds;seal=read(dest/'sealed.json');assert seal['queries']==COUNTS[ds]and not seal['GT_read'];keys=[]
 for i in range(seal['independent_shards']):
  sf=dest/'shards'/f'{i:05d}.json';j=read(sf);ack=read(sf.with_name(f'{i:05d}_audit.json'));assert ack['all_arrays_cold_exact']and ack['shard_sha256']==sha(sf);keys+=j['keys']
 assert len(keys)==len(set(keys))==COUNTS[ds]and set(keys)==set(seal['files'])
 for row in rows(ds):
  e=seal['files'][row['id']];assert sha(D/e['receipt_path'])==e['receipt_sha256'];a=arrays(e);control,ch=controls(ds,row['id']);assert ch==e['control_sha256']and all(np.array_equal(a[k],x)for k,x in control.items());cp=Path(e['compact_path']);cp=cp if cp.is_absolute()else D/cp;assert sha(cp)==e['compact_sha256']
  if e['origin']['kind']=='new_Source':
   dp=(D/e['origin']['native_path']).with_name('disposed.json');r=read(dp);assert r['native_Source_sha256']==e['origin']['native_sha256']and r['retained_full_grid_matching_input_sha256']==e['compact_sha256']
 atomic(dest/'input_audit.json',dict(status='all_full_domain_predictions_fullgrid_files_Source_reference_current_time_SHA_and_independent_shards_verified',queries=COUNTS[ds],seal_sha256=sha(dest/'sealed.json'),new_GT=0,worker=identity()));print(ds,'whole input audit complete',flush=True)
def metrics(b,ids,iv,pos,m,g,ds,manual=False,official_check=True):
 from stvg.metrics import official,box_iou
 from stvg.metrics_base import xyxy
 gt={int(f):x for f,x in g['boxes'].items()};ff=np.asarray(sorted(gt));truth=np.asarray([gt[int(f)]for f in ff],float);pix=xyxy(b,m['width'],m['height']);pix=np.maximum(pix,0)if ds=='hc2'else pix
 if manual:
  pix=pix/np.array([m['width'],m['height']]*2);truth=truth/np.array([m['width'],m['height']]*2);j=np.clip(np.searchsorted(ids,ff,side='right')-1,0,len(ids)-2);w=(ff-ids[j])/(ids[j+1]-ids[j]);pred=pix[j]*(1-w[:,None])+pix[j+1]*w[:,None];pred[ff<=ids[0]]=pix[0];pred[ff>=ids[-1]]=pix[-1]
 else:pred=np.column_stack([np.interp(ff,ids,pix[:,j])for j in range(4)])
 inter=np.maximum(np.minimum(pred[:,2:],truth[:,2:])-np.maximum(pred[:,:2],truth[:,:2]),0).prod(1);union=np.maximum(pred[:,2:]-pred[:,:2],0).prod(1)+np.maximum(truth[:,2:]-truth[:,:2],0).prod(1)-inter;iou=np.divide(inter,union,out=np.zeros(len(ff)),where=union>0);iou[(ff<ids[0])|(ff>ids[-1])]=0;a,z=map(int,iv);c,d=map(int,g['interval']);event=(ff>=max(a,c))&(ff<min(z,d));obs=np.isin(ff,ids[pos]);over=max(0,min(z,d)-max(a,c));t=over/(z-a+d-c-over);v=float(iou[event].sum()/max(max(z,d)-min(a,c),1))
 if not manual and official_check:
  off=official(b,dict(input=m,frame_ids=ids.tolist()),gt,g['interval'],iv,ds);assert abs(v-off['v'])<2e-10 and abs(t-off['t'])<2e-10;v,t=off['v'],off['t']
 fixed=[i for i,f in enumerate(ids)if int(f)in gt and i not in set(map(int,pos))];native=float(np.mean([box_iou(pix[i],np.asarray(gt[int(ids[i])])/np.array([m['width'],m['height']]*2)if manual else gt[int(ids[i])])for i in fixed]))if fixed else None
 distances=np.abs(ff[:,None]-ids[pos][None,:]).min(1)/m['fps'];bins=np.searchsorted([.5,1,2,4],distances,side='right');dr={}
 for j in range(5):mask=event&~obs&(bins==j);dr[str(j)]=dict(frames=int(mask.sum()),IoU=float(iou[mask].mean())if mask.any()else None)
 return dict(v=v,t=t,R03=float(v>.3),R05=float(v>.5),native_unobserved_IoU=native,event_unobserved_IoU=float(iou[event&~obs].mean())if(event&~obs).any()else None,event_observed_IoU=float(iou[event&obs].mean())if(event&obs).any()else None,event_unobserved_frames=int((event&~obs).sum()),event_observed_frames=int((event&obs).sum())),dr
def derive(ds,manual=False):
 assert read(D/ds/'input_audit.json')['seal_sha256']==sha(D/ds/'sealed.json');from studies.map_paper_completion_v1.common import labels
 truth=labels(ds);seal=fixed(D/ds/'sealed.json');prior={r['id']:r for r in fixed(OLD/'pure_full'/ds/'metrics.json')['private_rows']};oldmatch={r['id']:r for r in fixed(PILOT/ds/'metrics.json')['private_rows']};out=[];distance=[];checks=0
 for row in rows(ds):
  key=row['id'];e=seal['files'][key];a=arrays(e);g=truth[key];values={};dr={}
  for arm in ARMS:
   old=oldmatch[key]['values']['Matching']if arm=='Matching'and key in oldmatch else prior[key]['values'][arm]if arm!='Matching'else None;value,bins=metrics(a[arm],a['frame_ids'],a[arm+'_interval'],a['positions8'],row['metadata'],g,ds,manual,official_check=old is None)
   if old is not None:
    for field,key2 in [('v','v'),('t','t'),('native_unobserved_IoU','unobserved_actual_sIoU')]:
     if value[field]is None:assert old[key2]is None
     else:assert abs(value[field]-old[key2])<2e-10
   else:checks+=1
   if old is not None:value.update(v=old['v'],t=old['t'],R03=float(old['v']>.3),R05=float(old['v']>.5),native_unobserved_IoU=old['unobserved_actual_sIoU'])
   values[arm]=value;dr[arm]=bins
  assert len({x['t']for x in values.values()})==1;out.append(dict(id=key,parent=str(g['parent']),duration_group='short_le5s'if(g['interval'][1]-g['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s',values=values));distance.append(dict(id=key,parent=str(g['parent']),values=dr))
 return out,distance,checks
def score(ds):
 check();torch.set_num_threads(1);out,dr,n=derive(ds);atomic(D/ds/'metrics.json',dict(private_rows=out));atomic(D/ds/'distance_rows.json',dict(rows=dr));atomic(D/ds/'score_receipt.json',dict(status='whole_domain_fixed_time_matching_scored_original128_scalar_reuse_verified',queries=len(out),new_matching_official_checks=n,metrics_sha256=sha(D/ds/'metrics.json'),worker=identity()));print(ds,'whole official scoring complete',flush=True)
def score_audit(ds):
 check();torch.set_num_threads(1);want,dr,n=derive(ds,True);old=read(D/ds/'metrics.json')['private_rows'];distance=read(D/ds/'distance_rows.json')['rows'];maximum=0.
 for x,y,d,e in zip(want,old,dr,distance):
  assert all(x[k]==y[k]for k in ['id','parent','duration_group'])
  for arm in ARMS:
   for k,v in x['values'][arm].items():
    w=y['values'][arm][k];assert(v is None)==(w is None)
    if v is not None:maximum=max(maximum,abs(v-w));assert abs(v-w)<2e-10
   for bn,v in d['values'][arm].items():
    w=e['values'][arm][bn];assert v['frames']==w['frames']and(v['IoU']is None)==(w['IoU']is None)
    if v['IoU']is not None:assert abs(v['IoU']-w['IoU'])<2e-10
 atomic(D/ds/'score_audit.json',dict(status='whole_domain_independent_normalized_pixel_scores_distance_inputs_parents_verified',queries=len(old),max_scalar_error=maximum,metrics_sha256=sha(D/ds/'metrics.json'),distance_sha256=sha(D/ds/'distance_rows.json'),worker=identity()));print(ds,'whole independent score audit complete',flush=True)
def stats(rr,fields):
 if not rr:return dict(queries=0,parents=0,statistics={})
 parents=sorted({r['parent']for r in rr});lookup={p:i for i,p in enumerate(parents)};n=len(parents);s=np.zeros((n,len(fields)));c=np.zeros_like(s)
 for r in rr:
  for j,f in enumerate(fields):
   if r[f]is not None:s[lookup[r['parent']],j]+=r[f];c[lookup[r['parent']],j]+=1
 w=np.random.default_rng(20261022).multinomial(n,[1/n]*n,10000);den=w@c;boot=np.divide(w@s,den,out=np.full_like(den,np.nan),where=den>0);result={}
 for j,f in enumerate(fields):
  vv=[r[f]for r in rr if r[f]is not None];bb=boot[:,j];bb=bb[np.isfinite(bb)];result[f]=dict(mean=float(np.mean(vv))if vv else None,CI95=np.quantile(bb,[.025,.975]).tolist()if len(bb)else None,queries=len(vv),parents=int((c[:,j]>0).sum()))
 return dict(queries=len(rr),parents=n,statistics=result)
def report_data(ds):
 rr=read(D/ds/'metrics.json')['private_rows'];means={a:{k:float(np.mean(vv))if(vv:=[r['values'][a][k]for r in rr if r['values'][a][k]is not None])else None for k in rr[0]['values'][a]}for a in ARMS};paired={}
 for group in ['all','short_le5s','long_gt5s']:
  subset=[r for r in rr if group=='all'or r['duration_group']==group];paired[group]={}
  for a,b in PAIRS:
   dd=[]
   for r in subset:
    x,y=r['values'][a],r['values'][b];dv=100*(x['v']-y['v']);z=dict(parent=r['parent'],v= dv,t=100*(x['t']-y['t']),improved=100.*(dv>1e-10),harmed=100.*(dv< -1e-10),severe_v_gt5=100.*(dv< -5),rescue_v_gt5=100.*(dv>5))
    for f in ['native_unobserved_IoU','event_unobserved_IoU','event_observed_IoU']:z[f]=None if x[f]is None or y[f]is None else 100*(x[f]-y[f])
    dd.append(z)
   paired[group][a+'-'+b]=stats(dd,list(dd[0])[1:]if dd else[])
 distance={}
 for bn in map(str,range(5)):
  z=[]
  for r in read(D/ds/'distance_rows.json')['rows']:
   v={a:r['values'][a][bn]for a in ARMS};assert len({x['frames']for x in v.values()})==1
   if not v['Current']['frames']:continue
   z.append(dict(parent=r['parent'],**{a:100*v[a]['IoU']for a in ARMS},**{a+'-'+b:100*(v[a]['IoU']-v[b]['IoU'])for a,b in PAIRS}))
  distance[bn]=stats(z,list(z[0])[1:]if z else[])
 return dict(queries=len(rr),parents=len({r['parent']for r in rr}),means=means,paired=paired,conditional_dense_distance=distance,source_config=read(D/'plan.json')['source_config'][ds],scope='Full lists, same Source postfusiongrid, same8rawexpert softmasks/boxes/physicaltimes, same Full interval; no addedexpert/encoder. VideoWalk module adaptation, source functional but weak vs pure interpolation, not original learned-encoder reproduction. Parent10000 pointwiseCI exposedcohort, no target tuning. Native unobs covers all annotated native sampled unobserved frames; event_unobs is common predicted/GT dense intersection excludingexact8physicalobs; empty conditional null.')
def export(ds):
 check();assert read(D/ds/'score_audit.json')['metrics_sha256']==sha(D/ds/'metrics.json');rep=report_data(ds);atomic(P/ds/'report.json',rep);rr=read(D/ds/'metrics.json')['private_rows'];seal=read(D/ds/'sealed.json');cost=[]
 for i,row in enumerate(rows(ds),1):
  e=seal['files'][row['id']];cost.append(dict(query=f'Q{i:05}',kind=e['origin']['kind'],new_Source=e['new_Source'],new_expert=0,Source_seconds=e.get('Source_seconds'),matching_CPU_seconds=e.get('matching_CPU_seconds'),query_wall_seconds=e['query_wall_seconds'],worker=e['worker']))
 atomic(P/ds/'all_actual_query_costs.json',dict(rows=cost,inherited_Source_expert_cost_nonzero=True,cached_and_acquisition_clocks_not_deployment=True,nested_not_additive=True,counts=seal['counts']));pm={p:f'P{i:04}'for i,p in enumerate(sorted({r['parent']for r in rr}),1)}
 with(P/ds/'all_signed_scalars.tsv').open('w')as f:
  w=csv.writer(f,delimiter='\t',lineterminator='\n');fields=['v','t','R03','R05','native_unobserved_IoU','event_unobserved_IoU','event_observed_IoU'];w.writerow(['query','parent','duration_group']+[a+'_'+k for a in ARMS for k in fields])
  for i,r in enumerate(rr,1):w.writerow([f'Q{i:05}',pm[r['parent']],r['duration_group']]+[None if r['values'][a][k]is None else 100*r['values'][a][k]for a in ARMS for k in fields])
 with(P/ds/'all_signed_tails.tsv').open('w')as f:
  w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['query','comparison','delta_v_pp','decline_gt5pp','rescue_gt5pp'])
  for i,r in enumerate(rr,1):
   for a,b in PAIRS:v=100*(r['values'][a]['v']-r['values'][b]['v']);w.writerow([f'Q{i:05}',a+'-'+b,v,v< -5,v>5])
 text=[f'# {ds}: full same-Source-feature VideoWalk extension','',f'Complete {len(rr)} queries/{rep["parents"]}parent clusters; original128 outputs adopted without new prediction/Source/expert. All new grids retained; independent pre-GT source/array shards and wholeofficial/normalizedpixel/stat audits.','', '|Arm|vIoU|tIoU|R.3|R.5|Native unobs IoU|Conditional dense unobs IoU|','|---|---:|---:|---:|---:|---:|---:|']
 for a in ARMS:text.append('|'+a+'|'+'|'.join('null'if rep['means'][a][k]is None else f"{100*rep['means'][a][k]:.6f}"for k in ['v','t','R03','R05','native_unobserved_IoU','event_unobserved_IoU'])+'|')
 text+=['',rep['scope'],'','Current retains its original soft E/expert choices at observations; Matching always uses rawexpert boxes there. Same expert INPUT is not identical anchor OUTPUT. No Current hard-clamp or method/paper changes. All negative and severe outcomes/shortlong/support and distance CIs retained. Source recapture is only missing-input recovery, not a new expert or changed backbone. Experimental clocks include IO/cold audits and are not deployment latency.']
 for name,value in rep['paired']['all'].items():text.append(name+': '+json.dumps(value))
 (P/ds/'ANALYSIS.md').write_text('\n'.join(text)+'\n');atomic(D/ds/'reporting_receipt.json',dict(status='whole_domain_scored_allarray_normalizedpixel_parentmeansCI_signed_cost_exported_PENDING_ROOT',queries=COUNTS[ds],report_sha256=sha(P/ds/'report.json'),score_audit_sha256=sha(D/ds/'score_audit.json'),worker=identity()));print(ds,'whole signed results exported pending ROOT',flush=True)
def report_audit(ds):
 check();assert report_data(ds)==read(P/ds/'report.json');atomic(D/ds/'report_statistics_audit.json',dict(status='separate_process_allmeans_parentCI_distances_signed_summaries_rederived',report_sha256=sha(P/ds/'report.json'),worker=identity()));print(ds,'report statistical audit passed',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('task',choices=['bridge','shard_audit','whole_input_audit','score','score_audit','export','report_audit']);p.add_argument('--dataset',choices=COUNTS,required=True);p.add_argument('--shard',type=int);a=p.parse_args();globals()[a.task](*([a.dataset,a.shard]if a.task=='shard_audit'else[a.dataset]))
