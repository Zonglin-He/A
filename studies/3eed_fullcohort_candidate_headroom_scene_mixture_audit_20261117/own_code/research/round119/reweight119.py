"""Descriptive poststratification of fixed R118 outputs, no method selection.
All weights depend solely on full-cohort scene counts, never IoU or algorithm.
"""
from pathlib import Path
import json,hashlib,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round119';O=H/'analysis_attempt001';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 lock=rd(H/'experiment/POSTSTRATIFICATION_EXTENSION_LOCK.json')
 for p,h in lock['files'].items():assert sha(R/p)==h
 for p,h in lock['inputs'].items():assert sha(R/p)==h
 out=[];sr=[];weights=[]
 for src in ['waymo','drone','quad']:
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;full=rd(O/(name+'_ALL_ROWS_PRIVATE.json'));counts={sc:sum(r['scene_PRIVATE']==sc for r in full) for sc in sorted({r['scene_PRIVATE'] for r in full})};sets={co:rd(R/f'research/round118/scores_attempt001/{co}_{name}_ALL_ROWS_PRIVATE.json') for co in ['DEV','EXP']};sets['UNION']=sets['DEV']+sets['EXP'];assert len({r['id_PRIVATE'] for r in sets['UNION']})==len(sets['UNION'])
   for co,rows in sets.items():
    available=sorted({r['scene_PRIVATE'] for r in rows});covered=sum(counts[sc] for sc in available);mass=covered/len(full);ss={sc:[r for r in rows if r['scene_PRIVATE']==sc] for sc in available};pp={sc:counts[sc]/covered for sc in available}
    for sc in counts:weights.append({'cohort':co,'source':src,'target':tgt,'scene_PRIVATE':sc,'full_n':counts[sc],'cohort_n':len(ss.get(sc,[])),'full_fraction':counts[sc]/len(full),'weight_in_common_support':pp.get(sc,0.),'covered_full_mass':mass})
    for arm in rows[0]['arms']:
     vals=[]
     for sc in available:
      yy=np.array([r['arms'][arm]['IoU'] for r in ss[sc]]);row={'cohort':co,'source':src,'target':tgt,'scene_PRIVATE':sc,'arm':arm,'n':len(yy),'mIoU_percent':float(yy.mean()*100),'Acc25_percent':float((yy>.25).mean()*100),'Acc50_percent':float((yy>.5).mean()*100)};sr.append(row);vals.append(row)
     metrics={k:sum(pp[x['scene_PRIVATE']]*x[k] for x in vals) for k in ['mIoU_percent','Acc25_percent','Acc50_percent']};out.append({'cohort':co,'source':src,'target':tgt,'arm':arm,'n':len(rows),'covered_full_mass':mass,'scope':'FULL_SCENE_SUPPORT' if mass==1 else 'COMMON_SUPPORT_ONLY_NOT_FULL_ESTIMATE',**metrics})
 agg=[]
 for co in ['DEV','EXP','UNION']:
  for group in ['all_six_equal','waymo','drone','quad']:
   for arm in sets['DEV'][0]['arms']:
    mm=[x for x in out if x['cohort']==co and x['arm']==arm and (group=='all_six_equal' or x['target']==group)];agg.append({'cohort':co,'group':group,'arm':arm,'min_covered_full_mass':min(x['covered_full_mass'] for x in mm),'scope':'FULL_SCENE_SUPPORT' if all(x['covered_full_mass']==1 for x in mm) else 'HAS_COMMON_SUPPORT_ONLY_DIRECTION',**{k:float(np.mean([x[k] for x in mm])) for k in ['mIoU_percent','Acc25_percent','Acc50_percent']}})
 for n,x in [('R118_POSTSTRATIFIED_METHOD_ROWS',out),('R118_SCENE_SCALARS_PRIVATE',sr),('R118_POSTSTRATIFIED_AGGREGATE_ROWS',agg),('POSTSTRATIFICATION_WEIGHTS_PRIVATE',weights)]:dump(O/(n+'.json'),x)
 dump(H/'POSTSTRATIFICATION_RECEIPT.json',{'state':'COMPLETE_FIXED_R118_OUTPUTS_FULL_SCENE_FREQUENCY_POSTSTRATIFICATION','input_tasks':2304,'new_model_or_GT_scoring':0,'weight_depends_on_IoU':False,'selection_of_primary_or_parameters':False,'notes':'Within-scene sampling may remain biased; not fullcohort measurement or independent confirmation. EXP has only89/93 Waymo scenes, report common-supportscope; DEV andUNION coverall93/2/3.','FINAL_access':False});print('REWEIGHT119_COMPLETE',len(out),flush=True)
if __name__=='__main__':main()
