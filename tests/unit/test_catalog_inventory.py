"""Offline catalog discovery and atomic provenance regression tests."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace

import pytest
from llama_index.core.llms import MockLLM
from llama_index.core.schema import NodeWithScore, TextNode

from app.services.catalog_tables import validated_table_group
from app.services.corpus_html import parse_html
from app.services.lexical import is_inventory_question
from app.services.prepared_evidence import prepare_evidence, source_for_model
from app.services.rag import RAGConfig, RAGService
from app.services.related_evidence import RelatedEvidenceIndex


def catalog_fixture(tmp_path, *, rows=30):
    html = '<html><title>Widget</title><table><tr><td>Widget name</td><td>Assembly option</td><td>Applicability</td></tr>'
    html += '<tr><td rowspan="2">Shared α</td><td>Nonlinear δ</td><td>O</td></tr>'
    html += '<tr><td>Special ζ</td><td>X</td></tr>'
    html += ''.join(f'<tr><td>Widget {n} '+ 'literal ' * 7 + f'</td><td>Option {n}</td><td>O</td></tr>' for n in range(2, rows))
    html += '</table></html>'
    source = tmp_path / 'guide.html'
    source.write_text(html, encoding='utf-8')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    parsed = parse_html(source.read_bytes(), 'location')
    loc = {'source_id': 'source', 'source_hash': digest, 'location_id': 'location',
           'title': parsed['title'], 'product': 'civil', 'version': 'unknown'}
    catalog = {'sources': [{'source_id': 'source', 'source_hash': digest, 'format': 'html', 'relative_path': source.name}],
               'locations': [loc], 'blocks': [{**block, 'source_id': 'source', 'location_id': 'location'}
                                             for block in parsed['blocks']]}
    path = tmp_path / 'catalog.json'
    path.write_text(json.dumps(catalog), encoding='utf-8')
    return catalog, RelatedEvidenceIndex(tmp_path, path), path, source


def choose(resolver, query='List widget assembly options in CIVIL', **kwargs):
    diagnostics = {}
    group = resolver.inventory_table([query], product_versions=kwargs.pop('product_versions', {('civil', 'unknown')}),
                                     max_chars=kwargs.pop('max_chars', 6000), diagnostics=diagnostics, **kwargs)
    return group, diagnostics


@pytest.mark.parametrize('question', [
    'Какие варианты доступны в программе?', 'Перечисли категории элементов',
    'Which widget options are available?', 'List widget options', 'Give me a list of available widgets',
    'Расскажи, какие категории можно использовать в программе.',
    'Подскажи, пожалуйста, какие варианты доступны?', 'Скажи пожалуйста, какие варианты есть?',
    'Tell me which widget options are available', 'Could you tell me which categories can be used?',
])
def test_operation_detector_is_independent_of_topic(question):
    assert is_inventory_question(question)


@pytest.mark.parametrize('question', [
    'Что такое категория?', 'Как настроить список элементов?', 'Перечисли A и сравни с B',
    'List widgets and compare them', 'What is a widget?', 'How to configure options?',
    'Расскажи о категории', 'Tell me about a widget',
    'List widget assembly definitions', 'List widget options with configuration steps',
])
def test_mixed_definition_procedure_comparison_has_no_inventory_priority(question):
    assert not is_inventory_question(question)


def test_full_dom_group_keeps_row_zero_symbols_rowspan_and_identity(tmp_path):
    catalog, resolver, _, _ = catalog_fixture(tmp_path)
    group, report = choose(resolver)
    assert group is not None and report['selected_blocks'] == 31
    assert group['text'].startswith('Column 1=Widget name | Column 2=Assembly option')
    assert 'Column 1=Shared α | Column 2=Special ζ | Column 3=X' in group['text']
    assert group['text'].endswith('Column 2=Option 29 | Column 3=O')
    assert len(group['metadata']['block_ids']) == 31
    assert group['metadata']['atomic_evidence'] is True
    projected = source_for_model(group['metadata'])['block_attributes']
    assert projected['group_row_count'] == 31 and projected['structure'] == 'html_dom'
    counts = [json.loads(line) for line in projected['literal_column_counts'].splitlines()]
    assert next(row['counts'] for row in counts if row['column'] == 3) == {'Applicability': 1, 'O': 29, 'X': 1}
    assert len(projected['literal_column_counts']) <= 1600
    assert all(b['kind'] != 'table_header' for b in catalog['blocks'])
    assert choose(resolver)[0]['id'] == group['id']
    group['metadata']['block_ids'].clear()
    assert len(choose(resolver)[0]['metadata']['block_ids']) == 31


def test_heading_gate_rejects_incidental_row_value_and_product_only(tmp_path):
    _, resolver, _, _ = catalog_fixture(tmp_path)
    assert choose(resolver, 'List nonlinear algorithms in CIVIL')[0] is None
    assert choose(resolver, 'List CIVIL options')[0] is None
    assert choose(resolver, 'List widget assembly options')[0]


def test_semantic_navigation_can_bridge_languages_without_raising_evidence_score(tmp_path):
    _, resolver, _, _ = catalog_fixture(tmp_path)
    class Selector:
        def select(self, question, candidates, diagnostics):
            assert question == 'Перечисли способы сборки виджетов'
            assert 'Widget name' in candidates[0]['description']
            diagnostics['mode'] = 'fake'
            return candidates[0]['id']
    group, report = choose(resolver, 'List construction choices', semantic_selector=Selector(),
                           question='Перечисли способы сборки виджетов')
    assert report['anchored_candidates'] == 0
    assert group['metadata']['atomic_evidence']
    assert report['selected_blocks'] == 31
    assert report['semantic_selection']['mode'] == 'fake'


@pytest.mark.parametrize('mutation', ['source', 'catalog', 'choice'])
def test_semantic_navigation_revalidates_bytes_catalog_and_choice(tmp_path, mutation):
    _, resolver, path, source = catalog_fixture(tmp_path)
    class Selector:
        def select(self, question, candidates, diagnostics):
            if mutation == 'source':
                source.write_bytes(b'changed')
            if mutation == 'catalog':
                path.write_text('{}', encoding='utf-8')
            return 'invented' if mutation == 'choice' else candidates[0]['id']
    assert choose(resolver, 'List construction choices', semantic_selector=Selector())[0] is None


def test_semantic_navigation_never_receives_other_scope_or_overbudget_tables(tmp_path):
    _, resolver, _, _ = catalog_fixture(tmp_path)
    class Selector:
        def select(self, question, candidates, diagnostics):
            assert candidates == []
    assert choose(resolver, 'List construction choices', semantic_selector=Selector(), max_chars=10)[0] is None
    assert choose(resolver, 'List construction choices', semantic_selector=Selector(),
                  product_versions={('gts_nx', 'unknown')})[0] is None


@pytest.mark.parametrize('mutate', [
    lambda c: c['blocks'][2]['attributes']['resolved_cells'][0].update(text='invented'),
    lambda c: c['blocks'][2]['attributes']['cells'][0].update(rowspan=100),
    lambda c: c['blocks'][2].update(relations=[]),
    lambda c: c['blocks'][1]['attributes'].update(structure='pdf_extracted_unverified'),
    lambda c: c['sources'][0].update(format='pdf'),
    lambda c: c['locations'][0].update(source_hash='wrong'),
    lambda c: c['blocks'][2].update(location_id='different'),
])
def test_unverified_or_inconsistent_structure_fails_closed(tmp_path, mutate):
    catalog, _, _, _ = catalog_fixture(tmp_path)
    mutate(catalog)
    table = next(b for b in catalog['blocks'] if b['kind'] == 'table')
    rows = [b for b in catalog['blocks'] if b['kind'] in {'table_row', 'table_header'}]
    assert validated_table_group(table, rows, source=catalog['sources'][0], location=catalog['locations'][0]) is None


def test_budget_scope_staleness_and_catalog_invalidation(tmp_path):
    _, resolver, path, source = catalog_fixture(tmp_path)
    assert choose(resolver, product_versions={('gts_nx', 'unknown')})[0] is None
    assert choose(resolver, product_versions={('civil', '2021')})[0] is None
    group, report = choose(resolver, max_chars=100)
    assert group is None and report['bounded']
    assert report['omitted'][0]['reason'] == 'atomic_table_budget'
    resolver.warmup(inventory=True)
    view = resolver._table_view
    assert choose(resolver)[0] and resolver._table_view is view
    source.write_bytes(b'changed')
    assert choose(resolver)[1]['omitted'][0]['reason'] == 'stale_source'
    path.write_text('{}', encoding='utf-8')
    assert choose(resolver)[0] is None


def test_warm_view_is_shared_and_request_results_are_independent(tmp_path):
    _, resolver, _, _ = catalog_fixture(tmp_path)
    assert resolver.warmup(inventory=True)['catalog_loaded']
    view = resolver._table_view
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: choose(resolver)[0], range(8)))
    assert resolver._table_view is view
    assert len({g['id'] for g in results}) == 1
    results[0]['metadata']['block_ids'].clear()
    assert len(results[1]['metadata']['block_ids']) == 31


def test_duplicate_text_is_safe_but_competing_catalogs_fail_closed(tmp_path):
    catalog, _, path, source = catalog_fixture(tmp_path)
    html = source.read_text(encoding='utf-8')
    for replacement, expected in [('Widget 2', True), ('Widget X', False)]:
        raw = html + html.replace('Widget 2', replacement)
        source.write_text(raw, encoding='utf-8')
        parsed = parse_html(source.read_bytes(), 'location')
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        catalog['sources'][0]['source_hash'] = digest
        catalog['locations'][0]['source_hash'] = digest
        catalog['blocks'] = [{**b, 'source_id': 'source', 'location_id': 'location'} for b in parsed['blocks']]
        path.write_text(json.dumps(catalog), encoding='utf-8')
        group, report = choose(RelatedEvidenceIndex(tmp_path, path))
        assert bool(group) is expected
        if not expected:
            assert report['omitted'][0]['reason'] == 'ambiguous_table_anchor'


def test_product_partition_applies_before_table_top_k(tmp_path):
    catalog, _, path, _ = catalog_fixture(tmp_path)
    original = list(catalog['blocks'])
    for i in range(12):
        lid = f'other-{i}'
        catalog['locations'].append({**catalog['locations'][0], 'location_id': lid, 'product': 'gts_nx'})
        ids = {b['block_id']: f'{i}-{b["block_id"]}' for b in original}
        for row in deepcopy(original):
            row['location_id'] = lid
            row['block_id'] = ids[row['block_id']]
            if 'table_id' in row['attributes']:
                row['attributes']['table_id'] = ids[row['attributes']['table_id']]
            for rel in row.get('relations', []):
                rel['to_id'] = ids.get(rel['to_id'], rel['to_id'])
            catalog['blocks'].append(row)
    path.write_text(json.dumps(catalog), encoding='utf-8')
    group, _ = choose(RelatedEvidenceIndex(tmp_path, path))
    assert group and group['metadata']['product'] == 'civil'


def test_config_defaults_off_and_explicit_opt_in():
    assert RAGConfig.from_settings(SimpleNamespace()).catalog_inventory_enabled is False
    assert RAGConfig.from_settings(SimpleNamespace(rag_catalog_inventory_enabled=True)).catalog_inventory_enabled is True


@pytest.mark.parametrize('product', ['', 'unknown', 'unresolved', None])
def test_unknown_product_equality_does_not_establish_scope(tmp_path, product):
    catalog, _, path, _ = catalog_fixture(tmp_path)
    catalog['locations'][0]['product'] = product
    path.write_text(json.dumps(catalog), encoding='utf-8')
    resolver = RelatedEvidenceIndex(tmp_path, path)
    assert choose(resolver, product_versions={(product, 'unknown')})[0] is None


def rag_service(tmp_path, resolver, *, enabled=True, score=.9):
    config = RAGConfig(data_dir=tmp_path, docstore_path=tmp_path / 'docstore.json',
                       qdrant_url='http://unused.invalid', qdrant_api_key=None,
                       collection_name='inventory-test', baremetal_collection_name='inventory-baremetal',
                       llm_model='test', embedding_model='test', embedding_dimensions=3,
                       catalog_inventory_enabled=enabled, allow_external_data_transfer=True,
                       lexical_retrieval_enabled=False, min_score=.35, top_k=7)
    service = RAGService(config, llm=MockLLM(), client=SimpleNamespace())
    # Ordinary results deliberately contain only a parameter table/detail pages.
    # The inventory has no seed in the ordinary top-k and no invented edges.
    nodes = [NodeWithScore(node=TextNode(id_=f'ordinary-{i}', text=(f'parameter{i} ' * 300),
        metadata={'file_name': f'detail-{i}.html', 'product': 'civil', 'version': 'unknown',
                  'block_type': 'table_row' if i == 0 else 'paragraph'}), score=score) for i in range(7)]
    service._retriever = SimpleNamespace(retrieve=lambda _query: deepcopy(nodes))
    service._related_evidence = resolver
    return service


def test_retrieve_to_prepared_12k_keeps_atomic_catalog_and_stable_citations(tmp_path):
    _, resolver, _, _ = catalog_fixture(tmp_path)
    service = rag_service(tmp_path, resolver)
    first = service.retrieve('List widget assembly options in CIVIL', max_context_chars=12000)
    group_index = next(i for i, s in enumerate(first['sources']) if s['origin'] == 'catalog_inventory')
    group_source = first['sources'][group_index]
    expected = choose(resolver)[0]['text']
    assert len(expected) > 12000 / len(first['sources'])
    assert first['contexts'][group_index] == expected
    assert group_source['score'] == 0 and group_source['confident'] is False
    assert group_source['version'] == 'unknown'
    assert group_source['source_locator'] == 'guide.html'
    assert sum(map(len, first['contexts'])) <= 12000
    assert any(s['chunk_id'].startswith('ordinary-') for s in first['sources'])
    packet = prepare_evidence(first, query='List widget assembly options in CIVIL', budget=12000)
    assert packet['citation_evidence'][group_index]['text'] == expected
    again = service.retrieve('List widget assembly options in CIVIL', max_context_chars=12000)
    assert first['sources'][0]['chunk_id'] == again['sources'][0]['chunk_id']
    assert first['sources'][0]['source_hash'] == again['sources'][0]['source_hash']
    assert len(first['sources'][0]['block_ids']) == 31


def test_off_weak_retrieval_and_noninventory_preserve_ordinary_path(tmp_path):
    _, resolver, _, _ = catalog_fixture(tmp_path)
    for enabled, score, question in [(False, .9, 'List widget assembly options'), (True, .1, 'List widget assembly options'),
                                     (True, .9, 'How to configure widget assembly options?')]:
        result = rag_service(tmp_path, resolver, enabled=enabled, score=score).retrieve(question)
        assert not any(s.get('origin') == 'catalog_inventory' for s in result['sources'])


def test_off_branch_retains_ordinary_budget_allocations(tmp_path):
    _, resolver, _, _ = catalog_fixture(tmp_path)
    service = rag_service(tmp_path, resolver, enabled=False)
    question = 'List widget assembly options in CIVIL'
    legacy = service._prepare_answer(question, history=None, standalone_question=None, preserve_full_context=True).selection
    allocations = service._context_allocations(legacy.retrieved_contexts, 12000)
    expected = [service._bounded_context(text, size) for text, size in zip(legacy.retrieved_contexts, allocations, strict=True)]
    result = service.retrieve(question, max_context_chars=12000)
    assert result['contexts'] == [text for text in expected if text]
