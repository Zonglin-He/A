"""Retain the previous frozen Qwen-G baseline on the shared old336 only.
No fabricated source fallback for the64 new observations lacking this G cache.
"""
import numpy as np
from common25 import *
def main():
 out=BASE/'Qwen_control336_attempt001';out.mkdir(exist_ok=False);(out/'predictions_PRIVATE').mkdir();rr=[r for r in rows('T400') if r['stage']!='N'];assert len(rr)==336
 for r in rr:
  z=feature(r);folder=ROOT/'crossplatform/round24/bankM64_attempt001' if r['stage']=='M' else ROOT/'crossplatform/round23/bank272_attempt001';e=load_npz(folder/'samples_PRIVATE'/(r['id']+'.npz'));q=int((z['scores']+2*e['vision']+2*e['position']+2*e['relation']).argmax());np.savez_compressed(out/'predictions_PRIVATE'/(r['id']+'.npz'),query_index=np.array(q),box=z['boxes'][q])
 seal(out,'R25_old336_frozen_Qwen_G_control_preserved_before_current_scores',n=336,no_new_Qwen_calls=True,not_available_on_N64=True);print('QWEN_CONTROL336_SEALED')
if __name__=='__main__':main()
