import numpy as np
from common22 import *
def pref(v):
    out={}
    for candidate in ['A','B']:
        margins=[]
        for order in ['normal','reversed']:
            own=candidate if order=='normal' else ('B' if candidate=='A' else 'A');other='B' if own=='A' else 'A';x=v[order]['logits'];margins.extend([x[own]-x[other],x[own]-x['U']])
        out[candidate]=min(margins)
    return {'label':'A' if out['A']>=.5 else 'B' if out['B']>=.5 else 'U','A_min_margin':out['A'],'B_min_margin':out['B'],'blank_diagnostic_only':v['blank']['logits']}
def main():
    lock('ASSEMBLE_SCENE_LOCK.json');verify(BASE/'scene21_attempt001');out=BASE/'scene_direct21_attempt001';out.mkdir(exist_ok=False);(out/'predictions_PRIVATE').mkdir();(out/'cases_PRIVATE').mkdir()
    for r in read(EXP.parent/'input/SCENE21_PRIVATE.json'):
        z=feature(r);q0=int(z['scores'].argmax());selected={'source':q0,'scene_no_ref':q0,'scene_full':q0};v=read(BASE/'scene21_attempt001/cases_PRIVATE'/(r['id']+'.json'));pairs=old(r,'pairs')['pairs'];diag=[]
        for pair,p in zip(pairs,v['pairs']):
            assert pair['pair_alias']==p['pair_alias'];diag.append({'pair_alias':pair['pair_alias'],'qB_PRIVATE':pair['qB_PRIVATE'],'no_ref':pref(p['no_ref']),'full':pref(p['full'])})
        for arm,key in [('scene_no_ref','no_ref'),('scene_full','full')]:
            okay=[p for p in diag if p[key]['label']=='B']
            if okay:selected[arm]=max(okay,key=lambda p:(p[key]['B_min_margin'],-p['qB_PRIVATE']))['qB_PRIVATE']
        ix=np.array(list(selected.values()));np.savez_compressed(out/'predictions_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(list(selected)),query_indices=ix,arm_boxes_raw6=z['boxes'][ix]);dump(out/'cases_PRIVATE'/(r['id']+'.json'),{'alias':r['alias'],'pairs':diag})
    seal(out,'R22_label_enriched_scene_direct_predictions_sealed_before_score',n=21,optimizer_steps=0);print('SCENE_DIRECT22_SEALED')
if __name__=='__main__':main()
