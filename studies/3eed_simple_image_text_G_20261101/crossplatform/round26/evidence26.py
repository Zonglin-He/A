"""Frozen numeric full-subject and reference evidence; soft view aggregation only."""
import time,hashlib,numpy as np
from common26 import *
from observations26 import load
from visualZ import Evidence,crops
from reference21_v3 import crop as reference_crop
from association import DEFAULT_CONFIG
def main():
    from PIL import Image
    lock('EVIDENCE_LOCK.json');verify(BASE/'preparedF128_attempt001');verify(BASE/'dinoF128_attempt001');out=BASE/'evidenceF128_attempt001';out.mkdir(exist_ok=False);(out/'cases_PRIVATE').mkdir();plan=read(ROOT/'research/round15/experiment/SEMANTIC_PLAN_V4_LOCK.json');folder=Path(plan['model_dir']);assert all(sha(folder/f)==h for f,h in plan['model_files_sha256'].items());engine=Evidence(folder,DEFAULT_CONFIG);engine.setup_labels();calls=0;reused=0;start=time.perf_counter()
    for i,r in enumerate(read(EXP.parent/'input/F128_PRIVATE.json')):
        prep=read(BASE/'preparedF128_attempt001/cases_PRIVATE'/(r['id']+'.json'));g=read(BASE/'textF128_attempt001/cases_PRIVATE'/(r['id']+'.json'));d=read(BASE/'dinoF128_attempt001/cases_PRIVATE'/(r['id']+'.json'));payload={'alias':r['alias'],'targets_PRIVATE':{},'references':{}}
        if prep['mask_pool_PRIVATE']:
            frame,scene,binding=load(r);assert binding==prep['bindings'];oldv=old(r,'visual') if r['stage'] not in ('N','P') else {'masks_PRIVATE':{}};oldg=read(ROOT/'crossplatform/round21/text208_attempt003/cases_PRIVATE'/(r['id']+'.json')) if r['stage'] not in ('N','P') else {'references':[]};oldref=read(ROOT/'crossplatform/round21/reference208_attempt003/cases_PRIVATE'/(r['id']+'.json')) if r['stage'] not in ('N','P') else {'references':{}}
            if r['stage'] not in ('N','P'):assert hashlib.sha256(g['subject_query'].encode()).hexdigest()==oldv['full_subject_query_sha256']
            for m in prep['mask_pool_PRIVATE']:
                views=[]
                for mode,image in enumerate(crops(frame['rgb'],scene['objects'][m])):
                    cached=oldv['masks_PRIVATE'].get(str(m),[])
                    if cached:
                        x=cached[mode];assert arrsha(np.asarray(image))==x['full']['image_sha256'];full=x['full'];gray=x['gray_full'];reused+=2
                    else:
                        full=engine.score(image,g['subject_query']);gray=engine.score(Image.new('RGB',image.size,(127,127,127)),g['subject_query']);calls+=2
                    views.append({'full':full,'gray':gray,'visual_evidence':full['log_odds_yes_no']-gray['log_odds_yes_no']})
                payload['targets_PRIVATE'][m]=views
            for ref in g['references']:
                det=d['objects'][ref['entity']];obs=[];oldkey=next((x['entity'] for x in oldg['references'] if x['phrase']==ref['phrase'] and x['head']==ref['head']),None)
                for j in det['stable_indices'][:3]:
                    box=det['original']['boxes_PRIVATE'][j];cached=next((x for x in oldref['references'].get(oldkey,[]) if x['proposal_index']==j and np.array_equal(x['box_PRIVATE'],box)),None)
                    if cached:full=cached['full'];gray=cached['gray_full'];reused+=2
                    else:
                        image=reference_crop(frame['rgb'],box);full=engine.score(image,ref['phrase']);gray=engine.score(Image.new('RGB',image.size,(127,127,127)),ref['phrase']);calls+=2
                    obs.append({'proposal_index':j,'box_PRIVATE':box,'detector_score':float(det['original']['scores'][j]),'full':full,'gray':gray,'visual_evidence':full['log_odds_yes_no']-gray['log_odds_yes_no']})
                payload['references'][ref['entity']]=obs
        dump(out/'cases_PRIVATE'/(r['id']+'.json'),payload)
        if (i+1)%8==0:print('EVIDENCE23',i+1,'new',calls,'reused',reused,flush=True)
        assert calls<=5000,'Stop before unregistered expansion'
    assert engine.verify_unchanged();seal(out,'R23_full_subject_reference_soft_evidence_no_GT',n=128,new_calls=calls,reused_numeric_forwards=reused,model_unchanged=True,seconds=time.perf_counter()-start,peak_bytes=engine.torch.cuda.max_memory_allocated(),format_rejections=0);print('EVIDENCE23_SEALED',calls,reused)
if __name__=='__main__':main()
