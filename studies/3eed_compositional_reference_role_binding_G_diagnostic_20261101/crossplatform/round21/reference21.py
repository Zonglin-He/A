"""Numeric, role-specific verification of independently proposed reference boxes."""
import time
import numpy as np
from common21 import *
from observations import load
from visualZ import Evidence
from association import DEFAULT_CONFIG
def crop(rgb,box):
    from PIL import Image
    x1,y1,x2,y2=box;H,W=rgb.shape[:2];dx=.1*(x2-x1);dy=.1*(y2-y1);a,b=max(0,int(x1-dx)),max(0,int(y1-dy));c,d=min(W,int(x2+dx)),min(H,int(y2+dy));return Image.fromarray(rgb[b:d,a:c])
def main():
    from PIL import Image
    lock('REFERENCE_LOCK.json');verify(BASE/'dino208_attempt001');out=BASE/'reference208_attempt001';out.mkdir(exist_ok=False);(out/'cases_PRIVATE').mkdir();modelplan=read(ROOT/'research/round15/experiment/SEMANTIC_PLAN_V4_LOCK.json');modeldir=Path(modelplan['model_dir']);assert all(sha(modeldir/f)==h for f,h in modelplan['model_files_sha256'].items());engine=Evidence(modeldir,DEFAULT_CONFIG);engine.setup_labels();calls=0;start=time.perf_counter()
    for i,r in enumerate(read(ROWS)):
        g=read(BASE/'text208_attempt001/cases_PRIVATE'/(r['id']+'.json'));d=read(BASE/'dino208_attempt001/cases_PRIVATE'/(r['id']+'.json'));payload={'alias':r['alias'],'references':{},'not_identity_gold':True}
        if d['objects']:
            frame,scene,binding=load(r)
            for ref in g['references']:
                det=d['objects'][ref['entity']];observations=[]
                for j in det['stable_indices'][:3]:
                    box=det['original']['boxes_PRIVATE'][j];image=crop(frame['rgb'],box);blank=Image.new('RGB',image.size,(127,127,127));full=engine.score(image,ref['phrase']);head=engine.score(image,ref['head']);grayfull=engine.score(blank,ref['phrase']);grayhead=engine.score(blank,ref['head']);calls+=4
                    fv=full['log_odds_yes_no']-grayfull['log_odds_yes_no'];inc=fv-head['log_odds_yes_no']+grayhead['log_odds_yes_no'];accepted=full['log_odds_yes_no']>=.5 and fv>=.5 and (not ref['has_role_attributes'] or inc>=0)
                    observations.append({'proposal_index':j,'box_PRIVATE':box,'full':full,'head':head,'gray_full':grayfull,'gray_head':grayhead,'visual_evidence':fv,'modifier_increment':inc,'semantic_accepted':bool(accepted),'compositional_query':True,'single_view_not_identity_proof':True})
                payload['references'][ref['entity']]=observations
        dump(out/'cases_PRIVATE'/(r['id']+'.json'),payload)
        if (i+1)%8==0:print('REFERENCE21',i+1,'calls',calls,flush=True)
    assert engine.verify_unchanged();seal(out,'R21_reference_role_numeric_evidence_no_GT',n=208,calls=calls,model_unchanged=True,seconds=time.perf_counter()-start,peak_bytes=engine.torch.cuda.max_memory_allocated());print('REFERENCE21_SEALED',calls)
if __name__=='__main__':main()
