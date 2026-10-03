"""Descriptive post-seal top-1 corrections, same-population pairwise diagnostics."""
import sys,json,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_tastvg_large_evidence_public_v1 import independent_summary

def read(p):return json.loads(Path(p).read_text())
def key(r):return '/'.join(str(r[k]) for k in ['dataset','split','condition','order','arrival'])

def summarize(folder):
    folder=Path(folder);evidence={r['cell_key']:r for r in read(folder/'SCORE_ROWS.json')}
    scalarfolder=folder.parents[1]/'tastvg_large_correction_evidence'/'2026-10-03'
    scalar={r['cell_key']:r for r in read(scalarfolder/'EVIDENCE_ROWS.json')}
    output={};pairwise={};eps=1e-12
    for sp in ['search','confirm']:
        for ds in ['vidstg','hc2']:
            rows=read(folder/sp/ds/'ROWS.json')
            for cat in ['corruption','clean']:
                rr=[r for r in rows if r['expert_scheduled'] and (r['condition']=='clean')==(cat=='clean')]
                result={};pw=[]
                for arm in ['L8','L32','G8','G32']:
                    cc=collections.Counter();by_source=collections.defaultdict(collections.Counter)
                    for r in rr:
                        e=evidence[key(r)];a=e['anchor_index'];i=e['choices'][arm];distance=np.array(e['intervals'][i])-e['intervals'][a]
                        d=abs(distance).sum();b=2*abs(distance).min()/d if d>eps else 0.
                        if d<=eps:operation='same'
                        elif b<=.25+eps:
                            if abs(distance[0])>=abs(distance[1]):operation='trim_start' if distance[0]>0 else 'expand'
                            else:operation='trim_end' if distance[1]<0 else 'expand'
                        elif distance[0]<0<distance[1]:operation='expand'
                        elif distance[1]<0<distance[0]:operation='trim_both'
                        else:operation='shift'
                        large=d/(e['intervals'][a][1]-e['intervals'][a][0])>=.5-eps
                        gain=r[arm+'_gain'];cc['arrivals']+=1;cc['added_candidate_selected']+=i>=8
                        cc['changed']+=i!=a;cc['large_selected']+=large;cc['large_improved']+=large and gain>eps
                        cc['large_harmed']+=large and gain< -eps;cc['small_improved']+=(not large) and gain>eps
                        cc['small_harmed']+=(not large) and gain< -eps;cc['old_fast_gain_destroyed']+=r[arm+'_destroyed_old_fast']
                        by_source[str(r['source_id'])]['improved']+=gain>eps;by_source[str(r['source_id'])]['harmed']+=gain< -eps
                        cc['operation_'+operation]+=1
                        for threshold in [.3,.5]:
                            cc[f'v{threshold}_correct_destroyed']+=r['A8_v']>threshold>=r[arm+'_v']
                            cc[f'v{threshold}_correct_rescued']+=r[arm+'_v']>threshold>=r['A8_v']
                    result[arm]=dict(counts={k:int(v) for k,v in cc.items()},raw_source_counts={s:{k:int(v) for k,v in c.items()} for s,c in by_source.items()})
                for r in rr:
                    e=evidence[key(r)];scores={**e['scores'],**scalar[key(r)]['scores']}
                    for signal,z in scores.items():
                        for n in [8,32]:
                            cell={k:r[k] for k in ['source_id','condition','order','arrival']}
                            for kind,values in [('vIoU',r['candidate_v']),('tIoU',r['candidate_t'])]:
                                pairs=[(i,j) for i in range(n) for j in range(i+1,n) if abs(values[i]-values[j])>eps]
                                if pairs:
                                    credits=[.5 if abs(z[i]-z[j])<=eps else float((z[i]-z[j])*(values[i]-values[j])>0) for i,j in pairs]
                                    cell[kind+'_ordering']=float(np.mean(credits))
                            if 'vIoU_ordering' in cell and 'tIoU_ordering' in cell:pw.append(dict(**cell,signal=signal,support=n))
                name='/'.join([sp,ds,cat]);output[name]=result;pairwise[name]={}
                for signal in ['L','G','N','U','S']:
                    for n in [8,32]:
                        subset=[r for r in pw if r['signal']==signal and r['support']==n]
                        pairwise[name][signal+str(n)]=independent_summary(subset,['vIoU_ordering','tIoU_ordering'])
    (folder/'TOP1_DIAGNOSTICS.json').write_text(json.dumps(output,indent=2)+'\n')
    (folder/'PAIRWISE_DIAGNOSTICS.json').write_text(json.dumps(dict(scope='Post-seal descriptive only; compare signals within identical support. No target-score changes.',panels=pairwise),indent=2)+'\n')
    print(json.dumps({k:{s:v['counts'] for s,v in panel.items()} for k,panel in output.items() if k.startswith('confirm') and k.endswith('corruption')},indent=2))

if __name__=='__main__':summarize(sys.argv[1])
