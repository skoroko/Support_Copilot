"""Public Compose replay; fresh graphs, no Telegram/history/index writes.

Usage: python scripts/benchmark_prepared_evidence.py LABEL on|off CODE_ROOT
    AUDIT_MODE MODEL PREPARED_CHARS SEARCH_CHARS REASONING TABLE_SELECTION [IDS]
Runtime data/config remain at /app; CODE_ROOT may be an isolated candidate.
Optional IDS select public case IDs; the default runs the complete fixed eight.
Reports go to var/prepared_evidence_benchmark. No sealed holdout is loaded.
"""
import asyncio
import hashlib
import json
import logging
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'var' / 'prepared_evidence_benchmark'
ROOT.mkdir(parents=True, exist_ok=True)
label = sys.argv[1]
enabled = sys.argv[2] == 'on'
code_root = sys.argv[3] if len(sys.argv) > 3 else '/app'
sys.path.insert(0, code_root)
if code_root != "/app":
    sys.path.insert(1, "/app")
logging.disable(logging.CRITICAL)


async def main():
    import app.core.config as runtime_config
    # Relocated source must read exactly the same runtime paths/defaults.
    runtime_config.PROJECT_ROOT = Path("/app")
    runtime_config.DEFAULT_ENV_PATH = Path("/app/.env")
    get_settings = runtime_config.get_settings
    from app.observability.logging import setup_logging
    setup_logging('ERROR')
    from langgraph.checkpoint.memory import InMemorySaver

    from app.chat.agent import ChatAgent
    from app.chat.domain import Chat
    from app.chat.service import count_tokens
    from app.services.agent_persistent import TOOLS, build_agent
    from app.services.rag import RAGConfig, RAGService, build_llm_translate_hook
    from app.services.runtime_model import AgentModel

    settings = get_settings().model_copy(update={'rag_catalog_inventory_enabled': enabled,
        'rag_location_context_enabled': enabled,
        'agent_answer_audit': sys.argv[4] if len(sys.argv) > 4 else 'off'})
    if len(sys.argv) > 5:
        settings = settings.model_copy(update={'default_model': sys.argv[5]})
    if len(sys.argv) > 6:
        settings = settings.model_copy(update={'agent_prepared_context_chars': int(sys.argv[6]),
            'agent_search_context_chars': int(sys.argv[7])})
    if len(sys.argv) > 8:
        settings = settings.model_copy(update={'agent_reasoning_effort': sys.argv[8]})
    if len(sys.argv) > 9:
        settings = settings.model_copy(update={'rag_table_selection': sys.argv[9]})
    cases_path = Path(__file__).resolve().parents[1] / 'tests' / 'eval' / 'prepared_evidence_public_cases.json'
    cases = json.loads(cases_path.read_text(encoding='utf-8'))['items']
    if len(sys.argv) > 10 and sys.argv[10] == 'verification-52':
        cases_path = Path(__file__).resolve().parents[1] / 'tests' / 'eval' / 'verification_set.json'
        cases = json.loads(cases_path.read_text(encoding='utf-8'))['items']
        assert len(cases) == 52 and len({case['id'] for case in cases}) == 52
    elif len(sys.argv) > 10:
        cases = [case for case in cases if case['id'] in sys.argv[10].split(',')]
    config = replace(RAGConfig.from_settings(settings), ingest_on_start=False)
    def runtime_path(path):
        if path is None:
            return None
        path = Path(path)
        if code_root != "/app" and path.is_relative_to(Path(code_root)):
            return Path("/app") / path.relative_to(Path(code_root))
        return path if path.is_absolute() else Path("/app") / path
    config = replace(config, data_dir=runtime_path(config.data_dir),
                     docstore_path=runtime_path(config.docstore_path),
                     evidence_catalog_path=runtime_path(config.evidence_catalog_path),
                     visual_transcriptions_path=runtime_path(config.visual_transcriptions_path))
    start = time.perf_counter()
    rag = RAGService(config, translate_hook=build_llm_translate_hook(config), own_translate_hook=True).build()
    cache = rag._related_evidence.warmup(inventory=enabled)
    assert cache['catalog_loaded'] and cache['valid_sources'] > 0 and cache['invalid_sources'] == 0, 'runtime evidence paths mismatch'
    print(json.dumps({'catalog_loaded': cache['catalog_loaded'], 'valid_sources': cache['valid_sources'], 'table_groups': len(getattr(rag._related_evidence, '_table_groups', {})), 'label': label, 'stage': 'ready', 'startup_s': round(time.perf_counter()-start, 3), 'inventory_enabled': enabled, 'table_selection': getattr(config, 'table_selection', 'off'), 'table_selector_active': getattr(rag, '_table_selector', None) is not None}), flush=True)
    app_files = {path.relative_to(Path(code_root)).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted((Path(code_root) / 'app').rglob('*.py'))}
    manifest = {'label': label, 'app_sha256': hashlib.sha256(json.dumps(app_files, sort_keys=True).encode()).hexdigest(),
        'model': settings.default_model, 'reasoning_effort': settings.agent_reasoning_effort,
        'service_tier': settings.agent_service_tier, 'inventory_enabled': enabled,
        'location_context_enabled': enabled, 'answer_audit': settings.agent_answer_audit,
        'table_selection': settings.rag_table_selection,
        'prepared_chars': settings.agent_prepared_context_chars, 'search_chars': settings.agent_search_context_chars,
        'response_tokens': settings.chat_response_tokens, 'case_ids': [case['id'] for case in cases],
        'cases_sha256': hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        'catalog_sha256': hashlib.sha256(config.evidence_catalog_path.read_bytes()).hexdigest(),
        'fresh_in_memory_graph_per_case': True, 'telegram_sends_forbidden': True,
        'ingestion_enabled': False, 'case_retries': 0}
    (ROOT / (label + '.manifest.json')).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    count_tokens([{'role': 'user', 'content': 'warmup'}])
    calls, retrievals, rows, audit_inputs = [], [], [], []

    class TimedModel(AgentModel):
        async def ainvoke(self, *args, **kwargs):
            started = time.perf_counter()
            try:
                result = await super().ainvoke(*args, **kwargs)
            except Exception as error:
                calls.append({'seconds': round(time.perf_counter()-started, 3), 'error_type': type(error).__name__})
                raise
            calls.append({'seconds': round(time.perf_counter()-started, 3), 'usage': result.usage_metadata,
                          'service_tier': result.response_metadata.get('service_tier'),
                          'response_text': result.content,
                          'tool_calls': result.tool_calls})
            return result

        async def audit_answer(self, payload, *, mode):
            result = await super().audit_answer(payload, mode=mode)
            audit_inputs.append({'input': payload, 'result': result})
            return result

    original_retrieve = rag.retrieve
    def retrieve(*args, **kwargs):
        started = time.perf_counter()
        result = original_retrieve(*args, **kwargs)
        retrievals.append({'seconds': round(time.perf_counter()-started, 3), 'result': result})
        return result
    rag.retrieve = retrieve

    async def forbidden_send(*_args, **_kwargs):
        raise AssertionError('External sends are forbidden in this replay')

    model = TimedModel(settings, TOOLS)
    try:
        for case in cases:
            calls.clear()
            retrievals.clear()
            audit_inputs.clear()
            saver = InMemorySaver()
            graph = build_agent(saver, model=model, rag_provider=lambda: rag, send_message=forbidden_send)
            chat = Chat(owner_external_id='public-catalog-eval', interface='telegram')
            workers = set()
            def track(worker, workers=workers):
                workers.add(worker)
                worker.add_done_callback(workers.discard)
            agent = ChatAgent(graph, saver, settings=settings, track_worker=track)
            started = time.perf_counter()
            result, error_type = None, None
            try:
                async with asyncio.timeout(110):
                    result = await agent.turn(chat, case['question'], [])
            except Exception as error:  # noqa: BLE001 - retain failed benchmark rows without provider details
                error_type = type(error).__name__
            if workers:
                await asyncio.gather(*workers, return_exceptions=True)
            state = await graph.aget_state(agent.config(chat))
            answer = (result or {}).get('answer', '')
            sources = (result or {}).get('sources', [])
            markers = [int(number) for number in re.findall(r'\[(\d+)\]', answer)]
            summary = {'label': label, 'id': case['id'], 'seconds': round(time.perf_counter()-started, 3),
                       'error_type': error_type, 'model_calls': len(calls), 'retrievals': len(retrievals),
                       'answer_chars': len(answer), 'source_count': len(sources),
                       'citations_resolve': all(1 <= marker <= len(sources) for marker in markers)}
            rows.append({**summary, 'question': case['question'], 'answer': result,
                         'model_metrics': list(calls), 'retrieval_metrics': list(retrievals),
                         'evidence': state.values.get('citation_evidence') or [],
                         'audit_results': state.values.get('audit_results') or [],
                         'audit_inputs': list(audit_inputs)})
            (ROOT / (label+'.json')).write_text(json.dumps(rows, ensure_ascii=False, default=str, indent=2), encoding='utf-8')
            print(json.dumps(summary), flush=True)
    finally:
        await model.aclose()
        rag.close()


try:
    asyncio.run(main())
except Exception as error:  # noqa: BLE001 - safe diagnostic without credentials or provider payload
    print(json.dumps({'harness_error_type': type(error).__name__, 'module': getattr(error, 'name', None)}), flush=True)
    raise SystemExit(1)
