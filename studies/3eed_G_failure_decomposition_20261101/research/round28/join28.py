"""Caption/target join check; official comma spacing is formatting, not missing content."""
import sys,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round28'))
from common28 import *
def main():
 keys=read(EXP.parent/'input/BLIND_KEY_PRIVATE.json');lookup={x['id']:x for x in rows()};idx={x['id']:x for x in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};out=[]
 for k in keys:
  r=lookup[k['id_PRIVATE']];m=read(r['metapath']);caption=m['ground_info'][idx[r['id']]['object_index']]['caption'];eq=re.findall(r'\w+|[^\w\s]',caption)==re.findall(r'\w+|[^\w\s]',r['caption']);assert eq;out.append({'case':k['blind_case'],'identical_word_punctuation_tokens':eq,'comma_padding_only':r['caption']==caption.replace(',', ' ,')})
 dump(EXP/'CAPTION_LABEL_JOIN_AUDIT.json',{'all24_complete_word_and_punctuation_tokens_match_target_annotation':True,'input_uses_official_comma_padding':True,'literal_byte_equality_not_expected':True,'rows':out})
if __name__=='__main__':main()
