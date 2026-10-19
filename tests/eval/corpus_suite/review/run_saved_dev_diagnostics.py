import sys,json,time
from pathlib import Path
from collections import Counter,defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from app.eval.corpus_suite import read_json,write_new,digest,file_sha,slot_block_grounding,ExtractionVerifier
MAIN=Path(r'C:/00.MIDAS/PROJECTS/19. HelpDesk_Assistant/HelpDesk_Assistant')
SUITE=Path(__file__).resolve().parents[1]
OUT=SUITE/'review'
run_path=MAIN/'var/s0005_acceptance/dev_retrieval_v2_8.json'
run=read_json(run_path);contract=read_json(SUITE/'dev_v2_8/dev_reference_contract_v2.json');refs={r['case_id']:r for r in contract['cases']};cases=read_json(SUITE/'dev_v2_8/dev_cases_v2.json');by_case={c['id']:c for c in cases};contexts=read_json(SUITE/'dev_v2_8/dev_contexts_v2.json');norm=lambda s:''.join(s.split())
rows=[];patterns=Counter();pattern_cases=defaultdict(set)
for a in run['attempts']:
 if a['layer']!='retrieval':continue
 ref=refs[a['case_id']];records=(a.get('output') or {}).get('records',[])[:10]
 if a['context_sha256']!=contexts[a['case_id']]['fingerprint']:raise ValueError('Saved context mismatch')
 coverage=slot_block_grounding(ref,records);pending=ref['sufficiency']=='pending';scope='pending_reference_unscored' if pending else ('behavioral_reference_no_retrieval_ground_truth' if not ref['claim_slots'] else 'source_bound_scored')
 missing=[];goldhashes={e['source_hash'] for s in ref['claim_slots'] for alt in s['alternatives'] for e in alt['evidence']};actualhashes={r.get('source',{}).get('source_hash',r.get('source',{}).get('file_sha256')) for r in records}
 casepatterns=set()
 if len(goldhashes)>1 and goldhashes&actualhashes and goldhashes-actualhashes:casepatterns.add('missing_required_second_source')
 for slot,hit in zip(ref['claim_slots'],coverage['slots'],strict=True):
  if hit['matched']:continue
  for e in slot['alternatives'][0]['evidence']:
   samefile=[r for r in records if r.get('source',{}).get('source_hash',r.get('source',{}).get('file_sha256'))==e['source_hash']]
   sameloc=[r for r in samefile if r['source'].get('location_id')==e['location_id']]
   sameblock=[r for r in sameloc if e['block_id'] in {r['source'].get('block_id'),*(r['source'].get('block_ids') or [])}]
   quotehit=any(norm(e['quote']) in norm(r['text']) for r in sameblock)
   if quotehit:continue
   cause='required_source_absent' if not samefile else ('required_location_absent' if not sameloc else ('required_block_absent' if not sameblock else 'complete_literal_quote_absent'))
   casepatterns.add(cause)
   if e.get('source_review',{}):casepatterns.add('reviewed_visual_evidence_not_retrieved')
   if any(norm(e['quote']) in norm(r['text']) for r in records):casepatterns.add('literal_quote_present_under_other_identity_requires_review')
   missing.append({'slot_id':slot['slot_id'],'block_id':e['block_id'],'source_hash':e['source_hash'],'location_id':e['location_id'],'page':e['page'],'kind':e['kind'],'quote_sha256':e['quote_sha256'],'cause':cause})
 if scope=='source_bound_scored':
  for p in casepatterns:patterns[p]+=1;pattern_cases[p].add(a['case_id'])
 rows.append({'attempt_id':a['attempt_id'],'case_id':a['case_id'],'source_group':a['source_group'],'task_type':a['task_type'],'status':a['status'],'scope':scope,'attempt_sha256':digest(a),'contract_case_sha256':digest(ref),'legacy_metrics':a.get('metrics'),'curated_diagnostic':coverage,'quality_scored_metrics':{'recall':coverage['recall'],'all_slots':coverage['all_slots']} if scope=='source_bound_scored' else None,'missing_evidence':missing,'patterns':sorted(casepatterns)})
def summarize(rs):
 scored=[r for r in rs if r['scope']=='source_bound_scored'];mean=lambda vals:sum(vals)/len(vals) if vals else None
 return {'planned_attempts':len(rs),'statuses':dict(Counter(r['status'] for r in rs)),'scored_attempts':len(scored),'pending_reference_unscored':sum(r['scope']=='pending_reference_unscored' for r in rs),'behavioral_no_retrieval_ground_truth':sum(r['scope']=='behavioral_reference_no_retrieval_ground_truth' for r in rs),'curated_recall':{'value':mean([r['curated_diagnostic']['recall'] if r['status']=='completed' else 0. for r in scored]),'denominator':len(scored)},'curated_all_slots':{'value':mean([float(r['curated_diagnostic']['all_slots']) if r['status']=='completed' else 0. for r in scored]),'denominator':len(scored)},'legacy_recall_same_scored_denominator':{'value':mean([(r['legacy_metrics'] or {}).get('evidence_recall_at_10') or 0. for r in scored]),'denominator':len(scored)}}
report={'method':'offline_saved_attempt_strict_curated_rescore','original_run_file_sha256':file_sha(run_path),'original_run_sha256':digest(run),'reference_contract_sha256':digest(contract),'candidate_manifest_sha256':digest(read_json(SUITE/'dev_v2_8/manifest_v2.json')),'implementation_sha256':file_sha(Path(__file__)),'original_result_unchanged':True,'provider_calls':0,'semantic_success':None,'all_run_attempts':len(run['attempts']),'retrieval_attempts':len(rows),'summary':summarize(rows),'per_group':{g:summarize([r for r in rows if r['source_group']==g]) for g in sorted({r['source_group'] for r in rows})},'per_task_type':{g:summarize([r for r in rows if r['task_type']==g]) for g in sorted({r['task_type'] for r in rows})},'missing_patterns_scored_only':{p:{'attempts':n,'distinct_cases':len(pattern_cases[p])} for p,n in patterns.items()},'pending_case_ids':sorted({r['case_id'] for r in rows if r['scope']=='pending_reference_unscored'}),'attempts':rows,'release_status':'blocked; no semantic acceptance, no generation/e2e evaluation, no holdout'}
write_new(OUT/'dev_v2_8_all_retrieval_rescore.json',report)
print('retrieval',report['summary'],flush=True)
# Existing production catalog is loaded offline rather than rebuilding extraction.
import app.services
app.services.__path__.insert(0,str(MAIN/'app/services'))
from app.services.corpus import CorpusCatalog
index=MAIN/'var/s0005_acceptance/index';catalog=CorpusCatalog.load(index/'catalog.json');catalog_raw=read_json(index/'catalog.json');original_manifest=read_json(SUITE/'corpus_manifest.json');current_manifest=dict(original_manifest,docstore_sha256=file_sha(index/'docstore.json'))
verifier=ExtractionVerifier(MAIN/'data',index/'docstore.json',current_manifest,scratch_dir=OUT/'.reader-tmp-all90');verifier.visual_catalog=catalog
exrows=[]
for case in cases:
 started=time.perf_counter()
 try:output=verifier(case,contexts[case['id']],{});status='completed';error=None
 except Exception as exc:output=None;status='failed';error={'type':type(exc).__name__}
 locchecks=[]
 for a in case['required_evidence_slots']:
  locs=[l for l in catalog.locations if l.get('source_hash')==a['file_sha256'] and (a.get('physical_page') is None or l.get('page')==a['physical_page'])]
  quote=''.join(a['exact_quote'].split());matching=[]
  for l in locs:
   bs=[b for b in catalog.blocks if b['location_id']==l['location_id']]
   text=''.join(''.join(b.get('text','').split()) for b in bs)
   if quote in text:matching.append(l['location_id'])
  locchecks.append({'slot_id':a['slot_id'],'candidate_catalog_source_location_count':len(locs),'exact_anchor_quote_locations':matching,'expected_visual_regions':bool(a.get('expected_visual_regions'))})
 exrows.append({'case_id':case['id'],'source_group':case['source_group'],'task_type':case['task_type'],'status':status,'error':error,'verification_wall_ms':(time.perf_counter()-started)*1000,'scope':'no_source_anchor_not_applicable' if not case['required_evidence_slots'] else 'original_source_and_candidate_snapshot_verification','output':output,'catalog_anchor_checks':locchecks})
 if len(exrows)%15==0:print('extraction',len(exrows),flush=True)
checks=[x for r in exrows for x in (r.get('output') or {}).get('anchor_checks',[])]
summary={'cases':len(exrows),'statuses':dict(Counter(r['status'] for r in exrows)),'source_anchor_cases':sum(r['scope']!='no_source_anchor_not_applicable' for r in exrows),'no_anchor_not_applicable':sum(r['scope']=='no_source_anchor_not_applicable' for r in exrows),'passed_anchor_cases':sum(r['scope']!='no_source_anchor_not_applicable' and bool((r.get('output') or {}).get('extraction_pass')) for r in exrows),'anchor_checks':len(checks),'checks_passed':{k:sum(x[k] for x in checks) for k in ['snapshot_quote_match','independent_source_match','source_hash_match','required_visual_regions_verified']}}
write_new(OUT/'dev_v2_8_extraction_all90_index45.json',{'method':'offline_ExtractionVerifier_original_sources_and_index45','implementation_sha256':file_sha(Path(__file__)),'provider_calls':0,'holdout_opened':False,'catalog_file_sha256':file_sha(index/'catalog.json'),'docstore_file_sha256':file_sha(index/'docstore.json'),'visual_transcriptions_file_sha256':file_sha(index/'visual_transcriptions.json'),'original_corpus_manifest_sha256':digest(original_manifest),'effective_candidate_snapshot_manifest':current_manifest,'manifest_scope':'Source hashes retained from original corpus manifest; candidate docstore SHA substituted explicitly for the current candidate snapshot only. No original freeze modified.','visual_scope':'Uses existing45 catalog whose transcription bindings were independently checked offline; conservative all-image coverage on originally marked pages, not expert-selected required bbox completeness.','summary':summary,'cases':exrows})
print('extraction_summary',summary,flush=True)
