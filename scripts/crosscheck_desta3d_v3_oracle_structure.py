"""Root CPU readback of sealed rows and complete spectrum/projection identities."""
import sys,time,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vg_tta.desta3d_v3_oracle_io import read,write,sha,check_pins
S=ROOT/'artifacts/desta3d_v3/latent_oracle_v1/a02_global_mean_v1/structure_audit_v1'
def main():
    start=time.monotonic();assert read(S/'COMPLETE.json')['seal_sha']==sha(S/'SEAL.json')
    check_pins({str(S/k):v for k,v in read(S/'SEAL.json')['files'].items()});report=read(S/'PUBLIC_REPORT.json');maxerr=0.;checked=0
    for split,count in [('train',128),('dev',64)]:
        rows=[read(S/'rows'/split/f'{i:04}.json') for i in range(count)]
        for i,r in enumerate(rows):
            assert r['index']==i and r['shape'][-1]==256 and r['energy']>0
            m=r['metrics'];e=r['singular_value_squared'];assert len(e)==256
            assert all(x>=0 for x in e) and all(x>=y for x,y in zip(e,e[1:]))
            maxerr=max(maxerr,abs(math.fsum(e)/r['energy']-1))
            assert -1e-12<=m['global_energy']<=m['temporal_energy']+1e-12<=1+2e-12
            for kind in ('global','temporal'):
                if m[kind+'_cosine'] is not None:maxerr=max(maxerr,abs(m[kind+'_cosine']**2-m[kind+'_energy']))
            maxerr=max(maxerr,abs(m['temporal_extra_energy']-(m['temporal_energy']-m['global_energy'])))
            for k in (1,4,8,16,32):maxerr=max(maxerr,abs(math.fsum(e[:k])/r['energy']-m['rank%d_energy'%k]))
            for ax in r['neighbors'].values():assert ax['defined']+ax['undefined']==ax['pairs']
            checked+=1
        values={k:[r['metrics'][k] for r in rows] for k in rows[0]['metrics']}
        for axis in ('temporal','vertical','horizontal'):
            for metric in ('mean','median','difference_energy_over_total','mean_squared_difference_over_mean_token_energy'):
                values[axis+'_neighbor_'+metric]=[r['neighbors'][axis][metric] for r in rows]
            for k in ('pairs','defined','undefined'):
                assert sum(r['neighbors'][axis][k] for r in rows)==report['summary'][split]['statistics'][axis+'_neighbor_pair_counts'][k]
        for k,vals in values.items():
            v=[x for x in vals if x is not None];saved=report['summary'][split]['statistics'][k]
            assert saved['defined']==len(v) and saved['undefined']==len(vals)-len(v)
            if v:
                for key,x in [('mean',statistics.fmean(v)),('median',statistics.median(v)),('min',min(v)),('max',max(v))]:maxerr=max(maxerr,abs(x-saved[key]))
    assert checked==192 and maxerr<1e-10
    write(S/'ROOT_CROSSCHECK.json',dict(status='passed',queries=checked,max_identity_summary_error=maxerr,CPU_seconds=time.monotonic()-start,
        scope='All saved spectra/projection identities/neighbor counts/equal-query aggregates; raw NumPy vs Torch full-field calculation verified in primary audit. No native utility.'))
    print('ROOT_STRUCTURE_PASS',checked,maxerr)
if __name__=='__main__':main()
