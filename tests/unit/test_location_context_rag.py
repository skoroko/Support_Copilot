"""Physical packing preserves source facts and pre-existing linked evidence."""
from types import SimpleNamespace

from llama_index.core.llms import MockLLM
from llama_index.core.schema import NodeWithScore, TextNode

from app.services.rag import RAGConfig, RAGService, RetrievalPlan


def select(tmp_path, *, enabled=True, group_text='Seed fact.\nAdditional fact.', budget=18000, extra_groups=(), known_units=frozenset()):
    config = RAGConfig(data_dir=tmp_path, qdrant_url='http://unused.invalid', qdrant_api_key=None,
        collection_name='unit', baremetal_collection_name='bare', llm_model='test', embedding_model='test',
        embedding_dimensions=3, docstore_path=tmp_path / 'unused.json', lexical_retrieval_enabled=False,
        location_context_enabled=enabled)
    service = RAGService(config, client=SimpleNamespace(), llm=MockLLM())
    metadata = {'source_id': 's', 'source_hash': 'hash', 'location_id': 'p1', 'page': 1,
                'product': 'civil', 'version': 'unknown', 'file_name': 'guide.pdf', 'block_id': 'seed'}
    seed = NodeWithScore(node=TextNode(id_='seed-chunk', text='Seed fact.', metadata=metadata), score=.9)
    resolved = []
    def resolve(meta, text, **kwargs):
        resolved.append((meta['block_id'], text))
        return [{'block_id': 'note', 'text': 'Cross-page applicability condition.', 'metadata': {
            **metadata, 'location_id': 'p2', 'page': 2, 'block_id': 'note', 'origin': 'structural_relation'}}]
    service._related_evidence = SimpleNamespace(
        physical_contexts=lambda seeds, **_kw: [{'groups': [{'block_id': 'group', 'text': group_text, 'metadata': {
            **metadata, 'block_id': 'group', 'block_ids': ['seed', 'additional'],
            'origin': 'physical_location', 'atomic_evidence': True}}, *extra_groups], 'diagnostics': []} for _ in seeds],
        resolve=resolve, source_relative_path=lambda _: None)
    result = service._select_context(question='Seed fact', plan=RetrievalPlan('Seed fact', ('Seed fact',), 'standalone'),
        rankings=[[seed]], lexical_candidates=[], lexical_query=None, overview_question=False,
        preserve_full_context=True, inventory_budget=budget, known_atomic_units=known_units)
    return result, resolved


def test_completed_page_still_resolves_original_seed_cross_page_relations(tmp_path):
    result, resolved = select(tmp_path)
    assert resolved == [('seed', 'Seed fact.')]
    assert result.retrieved_contexts == ['Seed fact.', 'Cross-page applicability condition.', 'Seed fact.\nAdditional fact.']
    assert result.sources[-1]['origin'] == 'physical_location'
    assert result.sources[-1]['score'] == 0 and result.sources[-1]['confident'] is False
    assert result.top_score == .9


def test_additional_text_cannot_replace_or_change_original_seed(tmp_path):
    result, _ = select(tmp_path, group_text='Unrelated extracted detail. ' * 100)
    assert result.retrieved_contexts[0] == 'Seed fact.'
    assert result.sources[0]['origin'] != 'physical_location'


def test_disabled_completion_keeps_seed_and_original_relation(tmp_path):
    result, resolved = select(tmp_path, enabled=False)
    assert result.retrieved_contexts == ['Seed fact.', 'Cross-page applicability condition.']
    assert resolved == [('seed', 'Seed fact.')]


def test_twelve_k_has_no_expansion_and_matches_disabled_selection(tmp_path):
    original, _ = select(tmp_path, enabled=False, budget=12000)
    current, _ = select(tmp_path, budget=12000)
    assert current.sources == original.sources
    assert current.retrieved_contexts == original.retrieved_contexts


def test_explicit_larger_quota_retains_more_whole_locations_and_reports_omissions(tmp_path):
    groups = [{'block_id': f'unit-{i}', 'text': f'Additional fact {i}. ' * 150, 'metadata': {
        'source_id': 's', 'source_hash': 'hash', 'location_id': f'extra-{i}', 'block_id': f'unit-{i}',
        'product': 'civil', 'version': 'unknown', 'origin': 'physical_location', 'atomic_evidence': True,
        'source_locator': f'fictional.pdf#page={i + 10}'}} for i in range(7)]
    small, _ = select(tmp_path, extra_groups=groups)
    larger, _ = select(tmp_path, extra_groups=groups, budget=24000)
    old = [text for source, text in zip(small.sources, small.retrieved_contexts, strict=True)
           if source.get('origin') == 'physical_location']
    new = [text for source, text in zip(larger.sources, larger.retrieved_contexts, strict=True)
           if source.get('origin') == 'physical_location']
    assert len(old) < len(new) <= 6 and sum(map(len, old)) <= 6000 and sum(map(len, new)) <= 12000
    assert larger.retrieved_contexts[:2] == small.retrieved_contexts[:2]
    report = next(row for row in larger.related_evidence_diagnostics if row.get('origin') == 'physical_packing')
    assert report['budget_chars'] == 12000 and report['remaining_chars'] >= 0
    assert len(report['selected']) == len(new) and report['omitted']
    assert all(row['reason'] in {'location_limit', 'character_budget', 'duplicate_location'} for row in report['omitted'])


def test_retrieve_preserves_exact_prior_twelve_k_packet_before_bonus(tmp_path):
    from dataclasses import replace

    from app.services.prepared_evidence import evidence_identity

    config = RAGConfig(data_dir=tmp_path, qdrant_url='http://unused.invalid', qdrant_api_key=None,
        collection_name='unit', baremetal_collection_name='bare', llm_model='test', embedding_model='test',
        embedding_dimensions=3, docstore_path=tmp_path / 'unused.json', lexical_retrieval_enabled=False)
    service = RAGService(config, client=SimpleNamespace(), llm=MockLLM())
    texts = [('Fact X.\n' * 400) + str(i) for i in range(7)]
    sources = [{'chunk_id': str(i), 'file_name': 'guide.pdf', 'page': i + 1} for i in range(7)]
    selection = SimpleNamespace(sources=sources, retrieved_contexts=texts, has_evidence=True, ranked=[])
    service._prepare_answer = lambda question, **_kwargs: SimpleNamespace(selection=selection, telemetry={}, question=question)
    original = service.retrieve('Question', max_context_chars=12000)
    service.config = replace(config, location_context_enabled=True)
    selection.sources = [*sources, {'chunk_id': 'complete', 'file_name': 'guide.pdf', 'page': 8,
        'origin': 'physical_location', 'atomic_evidence': True}]
    selection.retrieved_contexts = [*texts, 'Complete unit.\n' * 200]
    current = service.retrieve('Question', max_context_chars=18000)
    assert current['contexts'][:7] == original['contexts']
    assert [evidence_identity(s) for s in current['sources'][:7]] == [evidence_identity(s) for s in original['sources']]
    assert current['contexts'][-1] == selection.retrieved_contexts[-1]
    assert sum(map(len, current['contexts'])) <= 18000
    selection.sources.extend([{'chunk_id': f'complete-{i}', 'file_name': 'guide.pdf', 'page': i + 10,
        'origin': 'physical_location', 'atomic_evidence': True} for i in range(3)])
    selection.retrieved_contexts.extend(['More complete text.\n' * 120 for _ in range(3)])
    larger = service.retrieve('Question', max_context_chars=24000)
    assert larger['contexts'][:7] == original['contexts']
    assert len(larger['contexts']) > len(current['contexts']) and sum(map(len, larger['contexts'])) <= 24000


def test_expanded_packet_keeps_short_sections_after_six_locations(tmp_path):
    groups = [{'block_id': f'short-{i}', 'text': f'Seed fact. Additional short section {i}.', 'metadata': {
        'source_id': 's', 'source_hash': 'hash', 'location_id': f'section-{i}', 'block_id': f'short-{i}',
        'product': 'civil', 'version': 'unknown', 'origin': 'physical_location', 'atomic_evidence': True}}
        for i in range(8)]
    result, _ = select(tmp_path, extra_groups=groups, budget=24000)
    physical = [source for source in result.sources if source.get('origin') == 'physical_location']
    assert len(physical) == 9
    assert result.retrieved_contexts[:2] == ['Seed fact.', 'Cross-page applicability condition.']
    report = next(row for row in result.related_evidence_diagnostics if row.get('origin') == 'physical_packing')
    assert report['remaining_chars'] > 0 and not report['omitted']


def test_known_whole_unit_frees_packing_only_for_exact_provenance_and_text(tmp_path):
    from app.services.prepared_evidence import atomic_unit_fingerprint

    original, _ = select(tmp_path)
    source = original.sources[-1]
    text = original.retrieved_contexts[-1]
    fingerprint = atomic_unit_fingerprint(source, text)
    assert fingerprint
    current, _ = select(tmp_path, known_units=frozenset({fingerprint}))
    assert current.retrieved_contexts == original.retrieved_contexts[:2]
    report = next(row for row in current.related_evidence_diagnostics if row.get('origin') == 'physical_packing')
    assert report['omitted'][0]['reason'] == 'already_visible' and report['remaining_chars'] == 6000
    for key in ('source_id', 'source_hash', 'location_id', 'block_id', 'product', 'version'):
        assert atomic_unit_fingerprint({**source, key: 'different'}, text) != fingerprint
    assert atomic_unit_fingerprint(source, text[:-1]) != fingerprint
    assert atomic_unit_fingerprint({**source, 'atomic_evidence': False}, text) is None
    assert atomic_unit_fingerprint({**source, 'source_hash': None}, text) is None
