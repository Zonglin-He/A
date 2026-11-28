"""Cross the interval pool with shortlisted rankers in a separate replay."""
import argparse

import numpy as np
import torch

import evaluate_ablation as evaluation
from rankers import Design
from train_ablation import BANKS,digest,fit,load_new,read,source_guard,stamp,write

FOLLOWUP = {
    'ExtentAnchor14': Design(pool='extent',depth=0,anchor=True),
    'ExtentAnchor8': Design(pool='extent',channels=8,depth=0,anchor=True),
    'ExtentAttention1': Design(pool='extent',depth=1),
}


def train():
    source_guard()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    file = BANKS/'features-seal.json'
    seal = read(file)
    assert seal['source_only'] and seal['stage']['code']==stamp()
    training,train_meta = load_new('train','extent',seal)
    validation,val_meta = load_new('val','extent',seal)
    assert not set(train_meta['recording'])&set(val_meta['recording'])
    assert np.all(val_meta['modes'][::5]==0)
    results = {}
    for name,design in FOLLOWUP.items():
        results[name] = fit(name,design,training,validation,val_meta['modes']!=0)
        print(name,results[name]['degraded'],flush=True)
    write(BANKS/'followup-training.json',dict(cards=results,source_only=True,
          code=stamp(('train_ablation.py','rankers.py','followup.py')),inputs={'generated':digest(file)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=('train','predict','score'))
    phase = parser.parse_args().phase
    if phase=='train':
        return train()
    evaluation.RUN /= 'extent-followup'
    evaluation.DESIGNS.update(FOLLOWUP)
    # This panel needs only the frozen baseline, the interval control and the new crosses.
    for name in tuple(evaluation.DESIGNS):
        if name not in (evaluation.CONTROL,'Extent',*FOLLOWUP):
            del evaluation.DESIGNS[name]
    evaluation.FIXED.clear()
    evaluation.NAMES = tuple(evaluation.DESIGNS)
    evaluation.PHASES += ('followup',)
    evaluation.FILES += ('followup.py',)
    return {'predict':evaluation.predict,'score':evaluation.score}[phase]()


if __name__=='__main__':
    main()
