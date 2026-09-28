"""Seal-first geometry, full raw normalization and source support audit."""
import sys,argparse,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vg_tta.desta3d_v3_oracle_io import *
from scripts.desta3d_v3_large_mask import ARMS,OLD,SMALL

def preflight(name):
 from scripts.score_desta3d_v2_source_task_control_v2 import controls
 paths=local_dependencies([Path(__file__),ROOT/'scripts/crosscheck_desta3d_v3_oracle_summary.py',ROOT/'tests/test_desta3d_v3_large_mask.py'])
 write(OUT/name/'SCORER_PREFLIGHT.json',dict(status='passed',controls=controls(),source_label_sha=sha(PANEL/'SOURCE_RECORDS.json'),
  source_GT_already_used_for_masks=True,pins={str(p):sha(p) for p in paths}))

def score(name):
 import numpy as np,torch
 from scripts.desta3d_v3_privileged_ptd_qualification import equal
 from scripts.score_desta3d_v2_reference_audit import score_tube_independently,summarize_parents,summarize_arm,paired_parent_bootstrap
 from vg_tta.external_evidence_metrics import tensor_metrics
 from scripts.audit_desta3d_v3_latent_oracle_raw import KL
 from vg_tta.desta3d_v3_free_actuation import native_targets,margin
 from vg_tta.desta3d_v2_prediction_contract import validate_prediction
 torch.set_num_threads(4)
 d=OUT/name;o=d/'independent_readback_v1';assert not o.exists();pre=read(d/'SCORER_PREFLIGHT.json')
 try:
  check_pins(pre['pins']);done,s=verify_seal(d);check_pins(read(d/'LOCK.json')['pins']);cfg=read(d/'CONFIG.json')
  assert done['predictions']==s['predictions']==80 and done['queries']==16 and done['optimizer_steps']==0
  assert not cfg['target_input'] and cfg['alpha']==.25 and cfg['ratios']=={'event':.087687,'spatial':.170316}
  rows=read(d/'INPUTS.json');assert len(rows)==len({r['source'] for r in rows})==16
  assert sha(PANEL/'SOURCE_RECORDS.json')==pre['source_label_sha']==cfg['source_label_sha']
  payload={a:[] for a in ARMS};identity=[];rawchecks=[];diagnostics=[];maxnorm=maxscaled=maxcast=maxrealized=maxrelative=0.;cast_endpoint_checks=0
  def load(p):return torch.load(p,weights_only=False,map_location='cpu')
  def norm(t):
   x=t.detach().float().numpy().astype(np.float64).reshape(-1);return float(np.sqrt(np.sum(x*x)))
  for i,row in enumerate(rows):
   ep=d/'episodes'/f'{i:02}';ident=read(ep/'INPUT.json');comp=read(ep/'COMPLETE.json');base=load(ep/'original.pt')
   assert ident['key']==comp['key']==row['key'] and comp['exact_state'] and comp['optimizer_steps']==0
   for f in ['BASELINE_REPLAY.json']:
    checks=read(ep/f);assert len(checks)>=10 and all(checks.values())
   oldbase=load(OLD/'episodes'/f'{i:02}'/'original.pt');assert equal(base['readout'],oldbase['readout']) and ident['support']==oldbase['support']
   identity.append(dict(source=row['source'],key=row['key'],negative_control=ident['negative_control']))
   endpoints=load(ep/'FULL_ENDPOINT_BASES.pt');stock_norm=norm(endpoints['stock']);case={}
   for arm in ARMS:
    p=load(ep/(arm+'.pt'));validate_prediction(p,len(row['input']['frame_ids']))
    assert p['key']==row['key'] and p['source']==row['source'] and p['support']==ident['support'] and p['preprocess']==ident['preprocess']
    assert p['adapter_sha']==cfg['adapter_sha'] and p['frame_ids']==row['input']['frame_ids'] and p['video_sha256']==row['input']['video_sha256']
    assert not p['decoder_GT_prefix'] and not p['target_read'] and p['optimizer_steps']==0
    payload[arm].append(p)
    if arm=='original':continue
    b,loc,which=arm.split('_');z=load(ep/(arm+'_DELTA.pt'));log=z['norm'];n=norm(z['raw']);sn=norm(z['matched'])
    small=load(SMALL/'episodes'/f'{i:02}'/(b+'_late_'+which+'_DELTA.pt'))
    assert torch.equal(z['raw'],small['raw']) and log['small_raw_exact_replay']
    err=abs(n-log['raw_norm']);maxnorm=max(maxnorm,err);assert err<1e-8
    target=cfg['ratios'][b]*stock_norm
    assert abs(log['requested_norm']-target)<1e-7 and abs(log['stock_norm']-stock_norm)<1e-7
    applicable=n!=0;assert log['applicable']==applicable
    target=target if applicable else 0.
    assert abs(log['target_norm']-target)<1e-7
    expected=(z['raw'].numpy()*np.float32(log['scale']));err=float(np.max(np.abs(expected-z['matched'].numpy())));maxscaled=max(maxscaled,err);assert err==0
    assert abs(sn-target)<max(1e-10,target*1e-6)
    base32=endpoints['baseline'][b].numpy();updated=np.add(base32,z['matched'].numpy(),dtype=np.float32)
    actual_diff=np.subtract(updated,base32,dtype=np.float32).astype(np.float64)
    realized=float(np.sqrt(np.sum(actual_diff*actual_diff)));maxrealized=max(maxrealized,abs(realized-log['realized_norm']))
    assert abs(realized-log['realized_norm'])<1e-7 and abs(realized-target)<=max(1e-10,target*1e-3)
    maxrelative=max(maxrelative,abs(realized/stock_norm-(cfg['ratios'][b] if applicable else 0.)))
    rawchecks.append(dict(key=row['key'],arm=arm,raw_norm=n,matched_norm=sn,target_norm=target,requested_norm=log['requested_norm'],
     realized_FP32_norm=realized,stock_norm=stock_norm,realized_ratio=realized/stock_norm,scale=log['scale'],applicable=applicable))
    effects=load(ep/(arm+'_CAST.pt'));inj={}
    for branch,e in effects.items():
     idx=e['indices'].numpy();diff=e['after'].float()-e['before'].float();cn=norm(diff);maxcast=max(maxcast,abs(cn-e['delta_l2']))
     assert abs(cn-e['delta_l2'])<1e-7 and len(idx)==len(set(idx.tolist()))==e['changed_elements']
     assert len(idx)==0 or (min(idx)>=0 and max(idx)<e['numel'])
     assert e['full_endpoint_reconstruction_saved'] and e['dtype']=='torch.bfloat16'
     # Independent integer round-to-nearest, ties-to-even; no torch casting here.
     def bf16_bits(v):
      bits=np.ascontiguousarray(v,dtype=np.float32).view(np.uint32)
      return ((bits+np.uint32(0x7fff)+((bits>>16)&1))>>16).astype(np.uint16).reshape(-1)
     bb=endpoints['baseline'][branch].numpy()
     before=bf16_bits(bb);after=bf16_bits(np.add(bb,z['matched'].numpy(),dtype=np.float32) if branch==b else bb)
     changed=np.flatnonzero(before!=after)
     assert np.array_equal(changed,idx)
     assert np.array_equal(before[idx],e['before'].view(torch.uint16).numpy())
     assert np.array_equal(after[idx],e['after'].view(torch.uint16).numpy())
     import hashlib
     assert hashlib.sha256(before.tobytes()).hexdigest()==e['baseline_sha']
     assert hashlib.sha256(after.tobytes()).hexdigest()==e['current_sha']
     cast_endpoint_checks+=2
     if branch!=b:assert e['changed_elements']==0 and e['current_sha']==e['baseline_sha']
     inj[branch]=dict(delta_norm=cn,changed=e['changed_elements'],numel=e['numel'],fraction=e['changed_elements']/e['numel'])
    same=p['readout']['spatial_reference_token_ids']==base['readout']['spatial_reference_token_ids'] and p['interval']==base['interval'] and p['positions']==base['positions']
    tids=set(base['time_distribution']['time_token_ids']);ref=lambda p:[x for x in p['readout']['spatial_reference_token_ids'] if x not in tids]
    same_semantic=ref(p)==ref(base);td=p['time_distribution']['endpoint_logits'];bd=base['time_distribution']['endpoint_logits']
    tc=KL(bd,td) if same_semantic and td is not None and bd is not None and td.shape==bd.shape else None
    cl=p['readout']['coordinate_logits'];bl=base['readout']['coordinate_logits'];cc=KL(bl,cl) if same and cl is not None and bl is not None else None
    if b=='spatial':assert equal(p['time_distribution'],base['time_distribution']) and p['event_completion']==base['event_completion']
    case[arm]=dict(same_spatial_support=same,same_semantic_reference=same_semantic,time_KL=tc,conditional_1001_coordinate_KL=cc,injection=inj,missing_native_injection=[b for b in ['event','spatial'] if b not in effects],format_ok=p['format_ok'],interval=p['interval'])
   diagnostics.append(dict(key=row['key'],arms=case))
  write(o/'PRE_SCORE_AUDIT.json',dict(status='passed',time=time.time(),predictions=80,optimizer_steps=0,source_GT_already_used_for_masks=True,
   scalar_raw_norm_max_error=maxnorm,scaled_tensor_max_error=maxscaled,sparse_cast_norm_max_error=maxcast,identities=identity,
   realized_FP32_norm_max_error=maxrealized,realized_ratio_max_error=maxrelative,full_BF16_endpoint_hash_checks=cast_endpoint_checks,
   limitation='Full baseline FP32+scaled delta allow independent integer BF16 rounding and both changed/unchanged endpoint hash checks; this does not establish task utility.'))
  labels={r['key']:r for r in read(PANEL/'SOURCE_RECORDS.json')};results={a:[] for a in ARMS};mx=marginerr=0.;marginchecks=0
  for i,row in enumerate(rows):
   for arm in ARMS:
    p=payload[arm][i];lab=labels[row['key']];v=score_tube_independently(p,lab);vv=tensor_metrics(p,lab)
    err=max(abs(v[k]-vv[k]) for k in ['vIoU','sIoU','tIoU']);mx=max(mx,err);assert err<1e-6
    margins={}
    for b in ['event','spatial']:
     logits=p['time_distribution']['endpoint_logits'] if b=='event' else p['readout']['coordinate_logits']
     if logits is None:margins[b]=None;continue
     try:targets,valid=native_targets(lab,p,b)
     except ValueError as e:margins[b]={'missing':str(e)};continue
     values=margin(logits,targets,valid);xx=logits.double().numpy()[valid.numpy()];yy=targets.numpy()[valid.numpy()];other=[]
     for vec,t in zip(xx,yy):other.append(float(vec[t]-max(v for j,v in enumerate(vec) if j!=t)))
     if not values:margins[b]={'missing':'no valid supervised native positions'};continue
     e=float(np.max(np.abs(np.array(values)-other)));marginerr=max(marginerr,e);marginchecks+=len(values);assert e==0
     margins[b]=dict(all=values,mean=float(np.mean(values)),positive=sum(x>0 for x in values),count=len(values))
    results[arm].append(dict(key=row['key'],source=row['source'],metrics=v,format_ok=p['format_ok'],interval=p['interval'],margins=margins))
  parents={a:summarize_parents(x) for a,x in results.items()};metrics=['vIoU','sIoU','tIoU']
  comparisons={a+'_minus_original':{m:paired_parent_bootstrap(parents[a],parents['original'],m) for m in metrics} for a in ARMS[1:]}
  correct={}
  for b in ['event','spatial']:
   eligible={r['source'] for r in identity if r['negative_control']['temporal' if b=='event' else 'spatial']['eligible_changed_support']}
   a=b+'_large_correct';bad=b+'_large_wrong'
   correct[a+'_minus_'+bad]=dict(eligible_parents=len(eligible),excluded_non_discriminating=[r['source'] for r in identity if r['source'] not in eligible],
    all_parents={m:paired_parent_bootstrap(parents[a],parents[bad],m) for m in metrics},
    eligible_only={m:paired_parent_bootstrap({p:v for p,v in parents[a].items() if p in eligible},{p:v for p,v in parents[bad].items() if p in eligible},m) for m in metrics})
  retention={}
  for m in ['vIoU','tIoU']:
   good={r['key'] for r in results['original'] if r['metrics'][m]>.5};retention[m]=dict(eligible=len(good),arms={a:dict(retained=sum(r['key'] in good and r['metrics'][m]>.5 for r in rr),lost=[r['key'] for r in rr if r['key'] in good and r['metrics'][m]<=.5]) for a,rr in results.items()})
  report=dict(status='source_large_mask_completed_audited',arms={a:dict(rows=rr,summary=summarize_arm(rr,parents[a])) for a,rr in results.items()},
   comparisons=comparisons,correct_minus_wrong=correct,retention=retention,diagnostics=diagnostics,raw_normalization=rawchecks,
   geometry_scalar_tensor_max_error=mx,margin_loop_max_error=marginerr,margin_values_checked=marginchecks,
   scope='Same16 exposed Vid train parents; same late direction, event .087687/spatial .170316 stock-merger norm, no optimizer/GTprefix/target. Two prior source outcomes supplied fixed magnitude. Three neutral event cases retain zero. Not generalization or OPD. SourceGT supplies mask. Descriptive10000 paired bootstrap seed20260927 no multiplicity correction.')
  write(o/'REPORT.json',report)
  lines=['# Large-magnitude source mask control','',report['scope'],'','|arm|tIoU%|sIoU%|vIoU%|','|---|---:|---:|---:|']
  for a in ARMS:
   v=report['arms'][a]['summary']['parent_macro'];lines.append('|'+a+'|'+'|'.join(f'{100*v[m]:.6f}' for m in ['tIoU','sIoU','vIoU'])+'|')
  for b,m in [('event','tIoU'),('spatial','sIoU')]:
   lines+=['',b+' primary eligible correct-minus-wrong:',str(correct[b+'_large_correct_minus_'+b+'_large_wrong']['eligible_only'][m])]
  lines+=['','All sources, wrong controls, margins, conditional KL, cast differences, native-good retention and >5pp tails retained in JSON. Merger norm matching is continuous, not BF16 difference matching.']
  (o/'REPORT.md').write_text('\n'.join(lines)+'\n');write(o/'COMPLETE.json',dict(status='scored',report_sha=sha(o/'REPORT.json'),geometry_comparisons=240,max_error=mx))
 except BaseException as e:
  write(o/'FAILURE.json',dict(error=repr(e),traceback=__import__('traceback').format_exc()));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','score']);p.add_argument('--name',required=True);a=p.parse_args();globals()[a.action](a.name)
