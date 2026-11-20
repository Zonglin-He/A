"""Exercise the public single-frame predictor against sealed sensor-space boxes."""
import numpy as np
import torch

from common import OUT, prior, catalog, digest, read, save
from scope183 import ordinary, frame
from crossplatform.ground_regions import Grounder


def error(x,y):
    delta=x-y
    delta[6]=(delta[6]+np.pi/2)%np.pi-np.pi/2
    return float(np.max(np.abs(delta)))


def run():
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    roster=ordinary();cat=catalog();root=prior.FULL/'predictions'
    arms=prior.old.ARMS.index('shared_cut')
    output=[]
    for source in prior.SOURCES:
        grounder=Grounder(source)
        for target,rows in roster.items():
            if target==source:continue
            folder=root/(source+'_to_'+target);receipt=read(folder/'SEALED.json')
            assert digest(folder/'all.npz')==receipt['all']
            with np.load(folder/'all.npz') as p:
                for active in (True,False):
                    ix=int(np.flatnonzero(p['registered']==active)[0]);inp=frame(rows[ix])
                    got=grounder.predict(**inp)
                    delta=error(got,p['completed'][ix,arms])
                    output.append(dict(source=source,target=target,form='relation' if active else 'fallback',error=delta))
                    print(output[-1],flush=True)
                    assert delta<1e-5
        row=next((r for r in cat['rows'] if r['scope']=='noun' and r['source']==source),None)
        if row is None:
            del grounder
            continue
        # Two descriptions may share the stored frame. Use the matching noun-panel input.
        from predict_nouns import inputs
        inp=next(v['frame'] for v in inputs() if v['source']==source and v['uid']==row['uid'])
        with np.load(row['prior']) as previous:expected=previous['completed'][arms]
        delta=error(grounder.predict(**inp),expected)
        output.append(dict(source=source,target=inp['target'],form='ordered',error=delta))
        print(output[-1],flush=True);assert delta<1e-5
        del grounder
    save(OUT/'entry_replay.json',dict(checks=output,records=len(output),largest_error=max(r['error'] for r in output),
         reference='sealed original observation outputs',target_labels=False,FINAL_access=False))


if __name__=='__main__':
    run()
