"""Numerical signal audit without GT or text export; never changes fitting."""
from pathlib import Path
import sys,math,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_temporal_hypothesis32_v1 import BASE,read,write,sha

def main():
    import torch,numpy as np
    rr=read(BASE/'ROSTER_LOCK.json')['records'];jm={j['case_id']:j for j in read(BASE/'INPUT_BARRIER.json')['jobs']};fm={f['case_id']:f for f in read(BASE/'FIT_BARRIER.json')['files']};rows=[]
    for r in rr:
        cid=r['probe_alias'];d=read(BASE/'teacher'/(cid+'.json'));j=read(ROOT/jm[cid]['path']);assert sha(ROOT/fm[cid]['path'])==fm[cid]['sha256'];f=torch.load(ROOT/fm[cid]['path'],map_location='cpu',weights_only=False)['fits']['HYP'];hs=j['selection']['hypotheses'];p=np.asarray(f['Native_conditional']);q=np.asarray(f['Teacher_conditional']);positive=q>0;kl=float(np.sum(q[positive]*np.log(q[positive]/p[positive]))) if len(p) else None;mass=[sum(math.exp(x) for x in e['raw_label_logprobs']) for e in d['evidence']];unc=[e['posterior'][2] for e in d['evidence']];sets=[set(h['sampled_physical_frame_ids']) for h in hs];overlap=len(sets[0]&sets[1])/len(sets[0]|sets[1]) if len(sets)==2 else None
        rows.append(dict(case_id=cid,dataset=r['dataset'],informative=d['informative'],conditional_Teacher_Native_KL=kl,allowed_three_label_mass=mass,conditional_unknown_probability=unc,sampled_frame_set_Jaccard=overlap,absolute_reward_difference=abs(d['evidence'][1]['reward']-d['evidence'][0]['reward']) if d['informative'] else None,Student_optimizer_steps=f['optimizer_steps']))
    write(BASE/'NUMERICAL_SIGNAL_AUDIT.json',dict(status='pass',count=32,rows=rows,not_calibrated_semantic_correctness=True,no_GT_reader=True,no_text_export=True,no_feedback_or_selection_change=True,GT_read=False))
if __name__=='__main__':main()
