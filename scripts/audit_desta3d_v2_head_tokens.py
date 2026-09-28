"""CPU-only complete token-category decomposition of saved projection evidence."""
import json,sys,time,re
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v2_p0 import read,sha
from scripts.score_desta3d_v2_aux_recovery import save_once
OUT=ROOT/'artifacts/desta3d_v2/tta_v2/source_head_projection_v1'

def run():
    reference=torch.load(OUT/'CPU_REFERENCE.pt',map_location='cpu',weights_only=False)
    tokpath=ROOT/'checkpoints/ParallelTubeDecoding-Qwen3-VL-4B/tokenizer.json'
    t=read(tokpath);vocab={v:k for k,v in t['model']['vocab'].items()}
    vocab.update({int(x['id']):x['content'] for x in t['added_tokens']})
    out={}
    for b in ['event','spatial']:
        a=reference['B1'][b];c=reference['saved_supervised3'][b];n=len(a['targets']);rows=[]
        for i in range(n):
            token=vocab[int(a['targets'][i])]
            kind='time' if re.fullmatch(r'<t\d+>',token) else ('coordinate' if re.fullmatch(r'<\d+>',token) else 'other')
            r={'position':int(a['positions'][i]),'target':int(a['targets'][i]),'text':token,'category':kind,'ntp':bool(a['ntp'][i])}
            for k in a['token']:
                r[k+'_delta']=float(c['token'][k][i]-a['token'][k][i])
            rows.append(r)
        groups={}
        for cat in ['time','coordinate','other','NTP','MTP','all']:
            selected=[r for r in rows if cat=='all' or r['category']==cat or (cat=='NTP' and r['ntp']) or (cat=='MTP' and not r['ntp'])]
            if not selected:continue
            g={'count':len(selected)}
            for k in ['ce_fp64_delta','ce_round_delta','ce_actual_fp64_delta','target_fp64_delta','lse_fp64_delta','target_actual_delta','lse_actual_fp64_delta']:
                x=np.array([r[k] for r in selected],dtype=np.float64)
                g[k]={'sum':float(x.sum()),'within_category_mean':float(x.mean()),'contribution_to_branch_mean':float(x.sum()/n)}
                if k.startswith('ce_'):g[k].update(down=int((x<0).sum()),up=int((x>0).sum()),equal=int((x==0).sum()))
            groups[cat]=g
        out[b]={'groups':groups,'tokens':rows}
    save_once(OUT/'ROOT_TOKEN_CATEGORY_READBACK.json',{'status':'completed','time':time.time(),'source':'saved full-vocabulary projection and tokens only',
        'tokenizer_sha':sha(tokpath),'all81tokens_preserved':True,'branches':out,'no_new_model_or_GT':True})
    print(json.dumps({b:v['groups'] for b,v in out.items()},indent=2))

if __name__=='__main__':run()
