"""Independent NumPy algebra/norm and integer BF16 readback of three chains."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from vg_tta.desta3d_v3_oracle_io import OUT,read,write,sha,check_pins
from scripts.desta3d_v3_privileged_ptd_qualification import B1

def norm(x):
    x=np.asarray(x,dtype=np.float64).reshape(-1);return float(np.sqrt(np.dot(x,x)))

def run(name):
    dest=OUT/name;done=read(dest/'COMPLETE.json');check_pins(done['pins']);check_pins(read(dest/'LOCK.json')['pins'])
    st=torch.load(B1,weights_only=False,map_location='cpu')['adapter'];r=read(dest/'REPORT.json');rows=read(dest/'INPUTS.json');cases=[];maxerr=0.
    cpu=read(OUT/'diagnosis_cpu_v1/REPORT.json')
    for j,(row,c) in enumerate(zip(rows,r['cases'])):
        d=torch.load(dest/'episodes'/f'{j:02}'/'LAYERS.pt',weights_only=False,map_location='cpu');b=c['branch'];base=d['original']
        old=OUT/'oracle001/episodes'/f"{row['original_index']:02}"
        masks=torch.load(old/'ORACLE_MASKS.pt',weights_only=False,map_location='cpu');wrong=torch.load(old/'WRONG_MASKS.pt',weights_only=False,map_location='cpu')
        w=st['out_proj_'+b+'.weight'].double().numpy();bias=st['out_proj_'+b+'.bias'].double().numpy();gate=c['stats']['original']['gate']
        out={};bname='temporal' if b=='event' else 'spatial'
        for arm,a in d.items():
            z=a['z'].numpy();p=a['proposal'].numpy();updated=a['updated'].numpy()
            expected=z.astype('float64')@w.T+bias;err=float(np.max(np.abs(expected-p)));maxerr=max(maxerr,err);assert err<2e-6
            bits=updated.copy().view(np.uint32);rounded=((bits+np.uint32(0x7fff)+((bits>>16)&1))>>16).astype(np.uint16)
            assert np.array_equal(rounded,a['cast'].view(torch.uint16).numpy())
            report={'projection_FP64_max_error':err,'integer_BF16_round_exact':True}
            if arm!='original':
                m=masks[b] if arm==bname else wrong[bname][b]
                factor=(.25+.75*m.numpy())[...,None];assert np.array_equal(z,base['z'].numpy()*factor)
                dz=z-base['z'].numpy();dp=p-base['proposal'].numpy();du=updated-base['updated'].numpy();dc=a['cast'].float().numpy()-base['cast'].float().numpy()
                beforecast=gate*(dz.astype('float64')@w.T)
                # Bias cancels analytically. FP32 projection and addition remain visible finite arithmetic.
                diff=float(np.max(np.abs(beforecast-gate*dp)))
                report.update(z_delta_relative=norm(dz)/max(norm(base['z'].numpy()),1e-30),
                  projection_delta_relative=norm(dp)/max(norm(base['proposal'].numpy()),1e-30),
                  gated_delta_relative_to_base_F=norm(gate*dp)/max(norm(base['updated'].numpy()-gate*base['proposal'].numpy()),1e-30),
                  continuous_delta_norm=norm(du),cast_delta_norm=norm(dc),cast_changed_fraction=float(np.count_nonzero(dc)/dc.size),
                  algebra_bias_cancellation_error=diff,GT_margin=cpu['cases'][row['original_index']]['arms'][arm]['margins'][b],
                  native_metrics=cpu['cases'][row['original_index']]['arms'][arm]['metrics'])
                assert diff<1e-8
            out[arm]=report
        cases.append({'key':row['key'],'branch':b,'gate':gate,'arms':out})
    report={'status':'passed','cases':cases,'projection_max_error':maxerr,'full_saved_layers':True,
      'interpretation':'Mask changes post-reader vectors and real BF16 tokens; GT margins may move in either direction. Ratios alone do not establish task root cause.',
      'new_native':0,'optimizer_steps':0,'target_read':False,'receipt':read(dest/'RECEIPT.json')}
    write(dest/'ROOT_RAW_READBACK.json',report);print('passed',maxerr)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);run(p.parse_args().name)
