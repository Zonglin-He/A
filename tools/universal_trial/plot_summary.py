"""Plot completed, sealed trial evidence; no new prediction or selection."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from stvg.runtime import sha
from stvg.tube_runner import write
from stvg.metrics import DenseTube

def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folders=['tubedetr_universal_trial','tubedetr_universal_diagnosis','tubedetr_universal_confidence'];seals={}
    for folder in folders:
        f=ROOT/'results'/folder/'final_seal.json';s=json.loads(f.read_text());assert s['status']=='complete'
        for name,h in s['files'].items():assert sha(ROOT/name)==h,name
        seals[folder]=sha(f)
    from tools.tubedetr_universal_trial import inputs
    p,cfg,rows=inputs();lp=ROOT/cfg['labels'];assert sha(lp)==cfg['labels_sha256'];labels={v['id']:v for v in map(json.loads,lp.read_text().splitlines())}
    out=ROOT/'outputs/tubedetr_universal_trial';ps=json.loads((out/'sealed.json').read_text());points=[]
    for q in p['query_ids']:
        e=ps['files'][q];assert sha(out/e['path'])==e['sha256'];t=labels[q];truth={int(k):v for k,v in t['boxes'].items()}
        with np.load(out/e['path'])as z:
            meta=dict(frame_ids=z['frame_ids'].tolist(),input=rows[q]['metadata']);m=lambda b:DenseTube(b,meta,truth,t['interval']).score(z['Strong_interval'])['v'];base=m(z['Source'])
            for i in range(1,21):points.append([z['initial_rewards'][i]-z['initial_rewards'][0],100*(m(z['initial_boxes'][i])-base)])
    data=np.array(points);fig,ax=plt.subplots(1,2,figsize=(11,4.6));bad=(data[:,0]>0)&(data[:,1]<0)
    ax[0].scatter(data[~bad,0],data[~bad,1],s=15,c='#4486ad',alpha=.65,label='Other candidates');ax[0].scatter(data[bad,0],data[bad,1],s=19,c='#ce643e',alpha=.8,label='Higher reward, lower vIoU')
    ax[0].axhline(0,color='#555555',lw=.8);ax[0].axvline(0,color='#555555',lw=.8);ax[0].set(xlabel='Reward change from this query\'s Source',ylabel='vIoU change from this query\'s Source (pp)',title='Actual parameter candidates: reward mismatch');ax[0].legend(fontsize=8,loc='lower right')
    a=json.loads((ROOT/'results/tubedetr_universal_diagnosis/summary.json').read_text());c=json.loads((ROOT/'results/tubedetr_universal_confidence/summary.json').read_text());v=[a['metrics'][n]['v']for n in ['SourceStrong','CandidateRerank','PolicyCenter','AllStartDirect']]+[c['metrics']['corrected']['v']]
    names=['Source\n+Strong','Candidate\nrerank','Policy\ncenters','21-start\nDirect','Confidence\nreadout']
    bars=ax[1].bar(np.arange(5),v,color=['#697787','#ce643e','#8a7caf','#4486ad','#a0a886']);ax[1].set_xticks(np.arange(5),names,fontsize=8);ax[1].set(ylabel='m_vIoU (%)',ylim=(0,23),title='16 exposed queries; fixed Strong time')
    for b,val in zip(bars,v):ax[1].text(b.get_x()+b.get_width()/2,val+.3,f'{val:.2f}',ha='center',fontsize=9)
    oracle=json.loads((ROOT/'results/tubedetr_universal_trial/summary.json').read_text())['candidate_oracle_m_vIoU'];ax[1].axhline(oracle,ls='--',lw=1,color='#666666');ax[1].text(.1,oracle+.4,f'Candidate oracle: {oracle:.2f} (diagnostic)',fontsize=8)
    fig.suptitle('TubeDETR generic-parameter prototype: interfaces work; reward not qualified',fontsize=11);fig.text(.5,.01,'All candidates are real decoder executions. Pilot comparisons have unequal compute and untuned Direct. Confidence readout adds no training.',ha='center',fontsize=8);fig.tight_layout(rect=[0,.04,1,.95])
    dest=ROOT/'results/tubedetr_universal_trial/diagnosis.svg';fig.savefig(dest);png=out/'diagnosis.png';fig.savefig(png,dpi=160);plt.close(fig)
    write(dest.with_name('figure_seal.json'),dict(status='complete_figure',source_final_seals=seals,figure_sha256=sha(dest),plot_code_sha256=sha(Path(__file__)),no_new_experiment=True))

if __name__=='__main__':run()
