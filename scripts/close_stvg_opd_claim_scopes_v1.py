"""Join genuine completed C scopes; never claim completion of remaining N/cost."""
import ast,hashlib,os,shutil,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha,status
from scripts.close_stvg_opd_p7_phase_v1 import archive
from scripts.publish_stvg_opd_actionability_audit_v1 import git,CHECKOUT,BRANCH,put,safe,receipt_metadata
BASE=ROOT/'artifacts/stvg_opd_claim_strengthening_v1'
OUT=ROOT/'results/stvg_opd_claim_strengthening/2026-10-13/complete_scope_closing'
def close():
    from scripts.stvg_opd_expanded_state_common_v1 import verify as expanded_verify
    from scripts.run_stvg_opd_deployment_latency_v1 import verify as latency_verify
    expanded_verify();latency_verify();bindings={};views={}
    for ns in ['expanded_state','deployment_latency']:
        final=read(BASE/ns/'FINAL_COMPLETION.json');v=read(BASE/ns/'ROOT_VISUAL_REVIEW.json')
        assert final['status']=='complete' and all(final[k] for k in ['actual_root','actual_PDF_pixel_view','actual_remote_publication','actual_archive'])
        assert v['status']=='pass' and v['actual_pixel_view']
        assert read(BASE/ns/'CLOSING_FINAL_GITHUB_RECEIPT.json')['status']=='pass'
        for relative,b in v['PDF_files'].items():
            assert sha(ROOT/relative)==b['sha256'] and (ROOT/relative).stat().st_size==b['bytes'];views[relative]=b
        for n in ['FINAL_COMPLETION.json','ROOT_CLOSING.json','ROOT_VISUAL_REVIEW.json','CLOSING_FINAL_GITHUB_RECEIPT.json','ARCHIVE_VERIFICATION_RECEIPT.json']:
            bindings[ns+'/'+n]=sha(BASE/ns/n)
    case=read(BASE/'aligned_cases/FINAL_COMPLETION.json');caseview=read(BASE/'aligned_cases/ROOT_VISUAL_REVIEW.json')
    assert case['status']=='complete' and caseview['status']=='pass' and case['actual_PDF_pages_viewed']==caseview['PDF_pages_actually_viewed']==3
    private='output/pdf/stvg_opd_claim_strengthening/aligned_observations_private.pdf'
    assert sha(ROOT/private)==case['private_PDF_sha256']==caseview['pdf_sha256']
    assert sha(BASE/'FINITE_VISUAL_CLOSING_GITHUB_RECEIPT.json')==case['finite_closing_publication_receipt_sha256']
    assert sha(BASE/'FINAL_PREPARATION_AND_VISUAL_ARCHIVE_RECEIPT.json')==case['archive_receipt_sha256']
    assert read(BASE/'FINITE_VISUAL_CLOSING_GITHUB_RECEIPT.json')['actual_remote_bytes_SHA256_Git_blob_tree_verified']
    views[private]=dict(sha256=sha(ROOT/private),bytes=(ROOT/private).stat().st_size,pages=3,private_not_for_public_export=True,prior_actual_pixel_view_reused=True)
    for n in ['aligned_cases/FINAL_COMPLETION.json','aligned_cases/ROOT_CLOSING_RECEIPT.json','aligned_cases/ROOT_VISUAL_REVIEW.json','FINITE_VISUAL_CLOSING_GITHUB_RECEIPT.json','FINAL_PREPARATION_AND_VISUAL_ARCHIVE_RECEIPT.json']:
        bindings[n]=sha(BASE/n)
    write(BASE/'ROOT_ALL_AUTHORIZED_C_SCOPES_READBACK.json',dict(status='pass',scope='actual all three original C requests: 905-parent Expanded-State, three sealed historical real qualitative cases, complete paired deployment timing',
        bindings=bindings,new_formal_fits=5430,expanded_parents=905,actual_qualification_GPU_fits=48+64,actual_timed_query_paths=128,
        qualitative_cases=3,actual_PDFs=3,actual_PDF_pages=5,new_GTC_scoring_or_model_calls_in_join=0,
        remaining_LN768_complete=False,remaining_cost32_complete=False,CPU_only=True,GT_read=False,time=time.time()))
    write(BASE/'ROOT_VISUAL_REVIEW.json',dict(status='pass',scope='joined verified prior three real case PDF pages and newly actually inspected expanded/timing pages; private real-case pixels stay excluded',
        actual_pixel_view=True,PDF_files=views,PDFs=3,pages=5,new_actual_pages_viewed=2,prior_actual_pages_viewed_reused=3,all_current_bindings_exact=True,CPU_only=True,GT_read=False,time=time.time()))
    note='**2026-10-13｜C原三项授权全部实际完成。** Expanded-State原905非开发历史parent/两order/5430新fits与全source/math/state/input/dense/bootstrap/root、两域全部匿名正负结果；原三真实Native/Before/After/GT对齐案例（既有有限真实FINAL/三页pixel view沿精确SHA复用，私人RGB/GT几何不公开）；原32query/64完整资格＋128实际Full/Frozen部署路径/全部exact output-state-writeback/root均完成，合三PDF五页真实view/匿名完整public/archive。Full−reset pooled两域CI跨零，Vid Full−fullcarry两order和pooled正、HC仍跨零，不升级状态因果或安全gate。现C scope是真正全部三项完成，不是整个S/N/cost/未来main完成。原N缺失LN-only768（两域各128/三固定order）、原32 audited cost及后续证据决定须继续实际串行，baseline剩余formal维持人类cancel/13747同身份T standby；无新方法/论文/CURRENT晋升。\n\n'
    p=ROOT/'docs/RESEARCH_HISTORY.md';text=p.read_text()
    for a in ['## 1. 当前状态：先读这一节\n\n','### 9.4 更新日志（新记录加在最上面）\n\n']:
        assert text.count(a)==1;text=text.replace(a,a+note,1)
    p.write_text(text);archive(BASE/'archive_all_C_actual_closing','ARCHIVE_ALL_C_CLOSING_RECEIPT.json')
    scope='complete all currently authorized original C Expanded-State/three historical real aligned cases/full paired deployment timing; remaining LN-only/audited cost/future decisions are separate'
    write(BASE/'ROOT_CLOSING.json',dict(status='complete',scope=scope,bindings=bindings,root_all_scopes_sha256=sha(BASE/'ROOT_ALL_AUTHORIZED_C_SCOPES_READBACK.json'),
        archive_sha256=sha(BASE/'ARCHIVE_ALL_C_CLOSING_RECEIPT.json'),actual_root=True,actual_PDF_pixel_view=True,actual_remote_publication=True,actual_archive=True,PDFs=3,pages=5,CPU_only=True,GT_read=True,time=time.time()))
    write(BASE/'FINAL_COMPLETION.json',dict(status='complete',scope=scope,root_closing_sha256=sha(BASE/'ROOT_CLOSING.json'),
        expanded_state_complete=True,aligned_cases_complete=True,paired_deployment_latency_complete=True,
        actual_root=True,actual_PDF_pixel_view=True,actual_remote_publication=True,actual_archive=True,
        original_three_C_items_complete=True,LN768_complete=False,cost32_complete=False,whole_followup_request_complete=False,
        baseline_remaining_formal_cancelled=True,new_gate_created=False,new_state_intervention_launched=False,production_promoted=False,CPU_only=True,GT_read=True,time=time.time()))
    status(BASE/'STATUS.json',dict(status='complete_all_authorized_C_scopes_remaining_N_cost_followup',C_complete=True,whole_followup_request_complete=False,CPU_only=True,GT_read=False,time=time.time()))
    print('C_ALL_THREE_ORIGINAL_SCOPES_ACTUAL_COMPLETE')
def stage():
    assert read(BASE/'FINAL_COMPLETION.json')['status']=='complete';assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
    base=git('rev-parse','HEAD');assert base==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)
    assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
    unrelated=git('ls-files','--others','--exclude-standard').splitlines();owned=set()
    def copy(p,rel):
        if p.suffix=='.json':safe(read(p))
        else:ast.parse(p.read_text())
        dest=CHECKOUT/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);assert sha(dest)==sha(p);owned.add(rel)
    copy(Path(__file__).resolve(),'scripts/close_stvg_opd_claim_scopes_v1.py')
    for n in ['ROOT_ALL_AUTHORIZED_C_SCOPES_READBACK.json','ROOT_VISUAL_REVIEW.json','ROOT_CLOSING.json','FINAL_COMPLETION.json','ARCHIVE_ALL_C_CLOSING_RECEIPT.json']:
        p=BASE/n;put(OUT/n,dict(original_private_receipt_sha256=sha(p),projection=receipt_metadata(read(p))));copy(OUT/n,str((OUT/n).relative_to(ROOT)))
    for ns in ['expanded_state','deployment_latency']:
        p=BASE/ns/'CLOSING_FINAL_GITHUB_RECEIPT.json';put(OUT/ns/'CLOSING_FINAL_GITHUB_RECEIPT.json',dict(original_private_receipt_sha256=sha(p),projection=receipt_metadata(read(p))));copy(OUT/ns/'CLOSING_FINAL_GITHUB_RECEIPT.json',str((OUT/ns/'CLOSING_FINAL_GITHUB_RECEIPT.json').relative_to(ROOT)))
    m=OUT/'SOURCE_MANIFEST.json';write(m,dict(status='pass',scope='genuine all-three-C-scope joining root/visual/public/archive closure only; all original full outcomes bound in individual immutable verified commits; excludes self hash',files={n:dict(sha256=sha(CHECKOUT/n),bytes=(CHECKOUT/n).stat().st_size) for n in sorted(owned)}));copy(m,str(m.relative_to(ROOT)))
    entries=[]
    for n in sorted(owned):
        p=CHECKOUT/n;d=p.read_bytes();entries.append(dict(path=n,bytes=len(d),sha256=sha(p),blob_sha=hashlib.sha1(b'blob '+str(len(d)).encode()+b'\0'+d).hexdigest()))
    with tempfile.TemporaryDirectory(prefix='whole-C-index-') as folder:
        env={**os.environ,'GIT_INDEX_FILE':str(Path(folder)/'index')};git('read-tree',base,env=env)
        for e in entries:
            h=git('hash-object','-w',str(CHECKOUT/e['path']));assert h==e['blob_sha'];git('update-index','--add','--cacheinfo','100644',h,e['path'],env=env)
        tree=git('write-tree',env=env)
    write(BASE/'CLOSING_PUBLIC_STAGE.json',dict(status='reviewed_staged',repository='Zonglin-He/A',branch=BRANCH,base_commit=base,expected_tree=tree,files=entries,file_count=len(entries),bytes=sum(e['bytes'] for e in entries),
        unrelated_untracked_preserved=unrelated,scope='actual all currently authorized three C scopes joined FINAL; complete expanded outcomes and timed results separately verified; not N/cost/whole followup',time=time.time()))
    print('C_ALL_SCOPES_CLOSING_PUBLIC_STAGE',len(entries),tree)
if __name__=='__main__':stage() if len(sys.argv)>1 and sys.argv[1]=='stage' else close()
