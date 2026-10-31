"""Global exposed-development policy selection; no sample-specific GT filter."""
from common import ROOT,EXP,read,dump,sha,verify,seal,check_plan
def main():
 check_plan();src=ROOT/'crossplatform/round17/G_variants64_attempt001';verify(src);gate=read(EXP/'A_scoring_attempt001/GATE.json');assert gate['G_variants_seal_sha256']==sha(src/'SEALED.json')
 out=ROOT/'crossplatform/round17/G_selected64_attempt001';out.mkdir(exist_ok=False);(out/'cases_PRIVATE').mkdir()
 for row in read(EXP.parent/'input/A64_PRIVATE.json'):
  old=read(src/'cases_PRIVATE'/(row['id']+'.json'));policy=gate['platforms'][row['platform']]['selected_policy'];pref=old['variants'][policy] if policy else {'status':'unusable','reason':'no_global_policy_passed_development_gate','positive_queries':[],'negative_queries':[]}
  dump(out/'cases_PRIVATE'/(row['id']+'.json'),{'id_PRIVATE':row['id'],'platform':row['platform'],'policy':policy,'preference':pref})
 seal(out,'R17_global_policy_selected_no_per_sample_GT_filter',n=64,A_gate_sha256=sha(EXP/'A_scoring_attempt001/GATE.json'),variant_seal_sha256=sha(src/'SEALED.json'));print('SELECTED_G',gate['platforms'])
if __name__=='__main__':main()
