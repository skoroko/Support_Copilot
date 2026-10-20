"""Late evidence must share a fresh-search quota instead of starving by order."""
from app.services.prepared_evidence import allocate_search_batch, evidence_identity


def source(name, **extra):
    return {'chunk_id': name, 'file_name': 'manual.txt', 'product': 'civil', **extra}


def test_long_first_hit_cannot_starve_short_late_fact():
    old = [{'source': source('old'), 'text': 'old fact. ' * 1200}]
    first, late = source('first'), source('late')
    texts, omitted = allocate_search_batch(old, [first, late],
        ['early fact. ' * 800, 'Late independent option exists.'], budget=18000)
    assert texts[evidence_identity(late)] == 'Late independent option exists.'
    assert sum(map(len, texts.values())) + len(old[0]['text']) <= 18000
    assert omitted and omitted[0]['not_citable'] is True


def test_atomic_table_and_each_ordinary_candidate_keep_independent_identity():
    table = source('grid', atomic_evidence=True)
    sources = [table, *[source(str(i)) for i in range(8)]]
    contexts = ['row\n' * 1250, *[f'Option {i} exists.\n' + 'body. ' * 200 for i in range(8)]]
    texts, omitted = allocate_search_batch([], sources, contexts, budget=6000)
    assert texts[evidence_identity(table)] == contexts[0]
    assert all(f'Option {i} exists.' in texts[evidence_identity(sources[i+1])] for i in range(8))
    assert sum(map(len, texts.values())) <= 6000
    assert len(omitted) == 8


def test_repeated_evidence_uses_only_increment_and_cannot_erase_old_facts():
    a, b = source('a'), source('b')
    old = [{'source': a, 'text': 'Original fact.'}]
    texts, _ = allocate_search_batch(old, [a, a, b],
        ['Original fact.', 'Another independent fact.', 'Late fact.'], budget=100)
    assert 'Original fact.' in texts[evidence_identity(a)]
    assert 'Another independent fact.' in texts[evidence_identity(a)]
    assert texts[evidence_identity(b)] == 'Late fact.'
    assert sum(map(len, texts.values())) <= 100


def test_unfittable_atomic_record_is_omitted_whole_with_navigation_hint():
    a = source('table', atomic_evidence=True)
    texts, omitted = allocate_search_batch([], [a], ['Literal heading\nrow\n' * 100], budget=10)
    assert texts[evidence_identity(a)] == ''
    assert omitted[0]['heading_hint'] == 'Literal heading'
    assert omitted[0]['not_citable'] is True
