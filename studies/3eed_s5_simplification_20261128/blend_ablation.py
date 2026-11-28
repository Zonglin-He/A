"""Keep the two geometry deletions separable within the compact ranker."""
import numpy as np
import torch

from train_ablation import BANKS,digest,fit,load_new,read,source_guard,stamp,write
from rankers import Design

BLEND = {
    'CompactExtent': Design(pool='extent',channels=8,depth=0,width=64,anchor=True),
    'CompactNo90': Design(pool='no90',channels=8,depth=0,width=64,anchor=True),
}


def main():
    source_guard()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    bank_file = BANKS/'features-seal.json'
    seal = read(bank_file)
    assert seal['source_only'] and seal['stage']['code']==stamp()
    cards = {}
    for name,design in BLEND.items():
        training,train_meta = load_new('train',design.pool,seal)
        validation,val_meta = load_new('val',design.pool,seal)
        assert not set(train_meta['recording'])&set(val_meta['recording'])
        assert np.all(val_meta['modes'][::5]==0)
        card = fit(name,design,training,validation,val_meta['modes']!=0)
        cards[name] = card
        print(name,card['degraded'],flush=True)
    write(BANKS/'blend-training.json',dict(cards=cards,source_only=True,
          code=stamp(('train_ablation.py','rankers.py','blend_ablation.py')),inputs={'generated':digest(bank_file)}))


if __name__=='__main__':
    main()
