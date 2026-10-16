"""Apply the pre-score development advancement rule exactly once."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_instance_evidence32_v1 import BASE,read,put,sha
def main():
    rule=read(BASE/'ADVANCEMENT_RULE_LOCK.json');cpu=read(BASE/'EXPERIMENT_CPU_COMPLETION.json');d=read(BASE/'EXPERIMENT_POSTSEAL_RESULTS.json');assert cpu['status']=='complete'and cpu['result_sha256']==sha(BASE/'EXPERIMENT_POSTSEAL_RESULTS.json')
    idx={(r['dataset'],r['arm']):r['mean']for r in d['summary']if r['metric']=='m_vIoU'};eps=1e-6;domains=('hc2','vidstg')
    eligible={a:all(100*(idx[ds,a]-idx[ds,b])>=-eps for ds in domains for b in ('E0','Temporal'))and any(100*(idx[ds,a]-idx[ds,'E0'])>eps for ds in domains)for a in ('E1','E2')}
    response=all(100*(idx[ds,'E1']-idx[ds,'E2'])>eps for ds in domains)
    chosen='E1'if eligible['E1']and response else'E2'if eligible['E2']else None
    put(BASE/'DEVELOPMENT_DECISION.json',dict(status='locked_for_confirmation'if chosen else'finite_modification_not_advanced',count=32,rule_sha256=sha(BASE/'ADVANCEMENT_RULE_LOCK.json'),result_sha256=sha(BASE/'EXPERIMENT_POSTSEAL_RESULTS.json'),evidence_eligibility=eligible,response_beats_original_both_domains=response,selected_recipe=chosen,GT_read=True,GT_scope='sealed development evaluation only; no candidate/anchor/query decisions',claim='finite development screen, not significance/generalization or universal instance-evidence impossibility',confirmation='pending fixed recipe execution'if chosen else'not triggered by locked evidence rule',cross_backbone='conditional on viable recipe and real compatible source; not executed',no_more_prompt_or_parameter_search=True,CURRENT_changed=False))
    print(json.dumps(read(BASE/'DEVELOPMENT_DECISION.json')))
if __name__=='__main__':main()
