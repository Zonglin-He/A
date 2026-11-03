"""Descriptive post-seal scalar analysis; never tunes or generates predictions."""
import json,hashlib,itertools,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];H=ROOT/'research/round52'
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 score=H/'score_dev336_attempt001';seal=read(score/'SEALED.json')
 for rel,h in seal['files'].items():assert sha(score/rel)==h
 audit=read(H/'saved_scalar_audit_attempt001/AUDIT_RESULT.json');assert audit['state'].startswith('PASS_')
 rows=read(score/'ALL_TASK_ROWS_PRIVATE.json');summary=read(score/'TASK_SUMMARY.json')['ALL336'];arms=list(rows[0]['IoU']);out=H/'analysis_attempt001';assert not out.exists();out.mkdir()
 table=[]
 for arm in arms:
  v=np.array([r['IoU'][arm] for r in rows]);v0=np.array([r['IoU']['source'] for r in rows]);d=v-v0
  record={'arm':arm,'balanced_mIoU_percent':summary['balanced_mIoU_percent'][arm],'drone_mIoU_percent':summary['drone'][arm]['mIoU_percent'],'quad_mIoU_percent':summary['quad'][arm]['mIoU_percent'],'gain_vs_source_pp':float(np.mean(d)*100),'improved':int((d>1e-10).sum()),'worsened':int((d< -1e-10).sum()),'unchanged':int((np.abs(d)<=1e-10).sum()),'query_changes':sum(r['queries'][arm]!=r['queries']['source'] for r in rows),'positive_mass_pp':float(np.maximum(d,0).sum()*100),'negative_mass_pp':float(np.minimum(d,0).sum()*100)}
  for t,name in [(.25,'25'),(.5,'50')]:
   record['rescue'+name]=int(((v0<=t)&(v>t)).sum());record['break'+name]=int(((v0>t)&(v<=t)).sum());record['balanced_Acc'+name+'_percent']=sum(summary[p][arm]['Acc'+name+'_percent'] for p in ['drone','quad'])/2
  table.append(record)
 transitions=[]
 for fam in ['new','old']:
  for cue in 'VPR':
   others=[c for c in 'VPR' if c!=cue]
   for mask in range(4):
    subset=''.join(c for i,c in enumerate(others) if mask>>i&1);after=''.join(c for c in 'VPR' if c in subset+cue)
    aa=fam+'_'+subset if subset else 'source';bb=fam+'_'+after
    dv=np.array([r['IoU'][bb]-r['IoU'][aa] for r in rows]);transitions.append({'family':fam,'added_cue':cue,'before':aa,'after':bb,'balanced_gain_pp':summary['balanced_mIoU_percent'][bb]-summary['balanced_mIoU_percent'][aa],'improved':int((dv>1e-10).sum()),'worsened':int((dv< -1e-10).sum()),'query_changes':sum(r['queries'][bb]!=r['queries'][aa] for r in rows),'rescue25':sum(r['IoU'][aa]<=.25 and r['IoU'][bb]>.25 for r in rows),'break25':sum(r['IoU'][aa]>.25 and r['IoU'][bb]<=.25 for r in rows)})
 scenes=[]
 for p in ['drone','quad']:
  for i,scene in enumerate(sorted({r['scene_PRIVATE'] for r in rows if r['platform']==p})):
   rr=[r for r in rows if r['platform']==p and r['scene_PRIVATE']==scene];scenes.append({'platform':p,'scene_alias':('D' if p=='drone' else 'Q')+'_S'+str(i+1).zfill(2),'n':len(rr),'mIoU_percent':{a:math.fsum(r['IoU'][a] for r in rr)/len(rr)*100 for a in arms}})
 rawstats=read(ROOT/'crossplatform/round52/predicted336_attempt001/CUE_STATS_PRIVATE.json');cue_stats={}
 for key in rawstats[0]['cue_statistics']:
  cue_stats[key]={}
  for field in rawstats[0]['cue_statistics'][key]:
   vals=np.array([r['cue_statistics'][key][field] for r in rawstats]);cue_stats[key][field]={'median':float(np.median(vals)),'mean':float(vals.mean()),'max':float(vals.max()),'nonzero_observations':int((vals>0).sum())}
 dump(out/'ARM_METRICS.json',table);dump(out/'ALL_FACTORIAL_TRANSITIONS.json',transitions);dump(out/'ALL_SCENE_METRICS.json',scenes);dump(out/'CUE_SCALE_DIAGNOSTICS.json',cue_stats)
 dump(out/'ANALYSIS_RECEIPT.json',{'state':'R52_DESCRIPTIVE_POSTSEAL_NO_TUNING','n':336,'arms':len(arms),'transitions':len(transitions),'scene_clusters':len(scenes),'all_cases_and_negatives_retained':True,'semantic_identity_gold_available':False,'independent_test':False,'predictions_changed':False,'new_model_forwards':0,'TTA_steps':0,'inputs_sha256':{str(x.relative_to(ROOT)):sha(x) for x in [score/'SEALED.json',H/'saved_scalar_audit_attempt001/AUDIT_RESULT.json',ROOT/'crossplatform/round52/predicted336_attempt001/SEALED.json']}})
 print(json.dumps({'state':'DESCRIPTIVE_ANALYSIS_COMPLETE','arms':len(table),'transitions':len(transitions),'scenes':len(scenes)}))
if __name__=='__main__':main()
