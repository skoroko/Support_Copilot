"""Frozen-index evidence must keep its product, coverage and table meaning."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from app.services.evidence_text import behavior_context, chm_context, project_table_rows
from app.services.rag import RAGConfig, RAGService, RetrievalPlan


def test_behavior_evidence_does_not_classify_models_from_applicability_grid():
    prose = "Материалы возвращаются к исходной форме. Доступны линейные и нелинейные упругие модели."
    grid = "Таблица 4.1.1 Применение моделей материалов в зависимости от типа конечного элемента Model A √ √ Model B √"
    text = prose + " " + grid
    assert behavior_context(text, "Что означает упругое поведение и какие упругие модели доступны?") == prose
    assert behavior_context(text, "Какие упругие модели применимы для типа конечного элемента?") == text
    assert behavior_context(text, "Какие модели поддерживает балочный element?") == text
    # A parameter table does not have the compatibility-table semantics.
    parameters = prose + " Таблица 4.1.2 Параметры модели E=100"
    assert behavior_context(parameters, "Упругое поведение материала") == parameters


def test_behavior_projection_is_used_for_selected_pdf_context():
    service = RAGService(RAGConfig.from_settings(SimpleNamespace()))
    node = NodeWithScore(node=TextNode(
        text="Elastic definition. Table 1.1 Application of material models by element type Model A √",
        metadata={"file_name": "Materials.pdf"},
    ), score=0.7)
    assert service._context_text(node, question="What is elastic behavior?") == "Elastic definition."
    assert "Model A" in service._context_text(node, question="Material compatibility by element type?")


def test_translated_query_keeps_direct_monolingual_evidence():
    service = RAGService(RAGConfig.from_settings(SimpleNamespace()))
    query = 'стадии строительства накопительный грунта начальное состояние stages construction accumulated soil initial state'
    direct = NodeWithScore(node=TextNode(text='Construction stages: results are accumulated from previous stages. Soil initial state.'), score=0.54)
    generic = NodeWithScore(node=TextNode(text='стадии строительства грунта начальное состояние soil initial state'), score=0.60)
    nodes = [generic, direct]
    assert service._rank_by_lexical_evidence(query, nodes)[0] is direct
    service.config = replace(service.config, lexical_term_weight=0, lexical_pair_weight=0)
    assert service._rank_by_lexical_evidence(query, nodes)[0] is generic


def test_short_embedded_model_name_does_not_change_lexical_scoring():
    assert not RAGService._mixed_query_groups(RAGService._lexical_terms('расскажи про модель грунта Modified Mohr Coulomb'))


def test_standalone_translation_adds_only_one_bounded_dense_variant():
    service = RAGService(RAGConfig.from_settings(SimpleNamespace()))
    question = 'стадии строительства накопительный грунта stages construction accumulated soil initial state'
    plan = service._retrieval_plan(question, None, question)
    assert plan.queries == (question, 'stages construction accumulated soil initial state')
    short = 'модель упругости грунта Modified Mohr Coulomb'
    assert service._retrieval_plan(short, None, short).queries == (short,)


def test_translated_query_keeps_numeric_and_alphanumeric_anchors():
    service = RAGService(RAGConfig.from_settings(SimpleNamespace()))
    query = 'ошибка расчета модели boundary stiffness mesh solver convergence 4021 3d'
    groups = service._mixed_query_groups(service._lexical_terms(query))
    assert all({'4021', '3d'} <= group for group in groups)
    other = NodeWithScore(node=TextNode(text='boundary stiffness mesh solver convergence 4022 2d'), score=0.60)
    direct = NodeWithScore(node=TextNode(text='boundary stiffness mesh solver convergence 4021 3d'), score=0.59)
    assert service._rank_by_lexical_evidence(query, [other, direct])[0] is direct


@pytest.mark.parametrize(('text', 'expected'), [
    ('Chapter 4. Mesh | 65\nGeoTechnical analysis System\n0\nParameters', 65),
    ('Chapter 5. Analysis Method | 7\nCumulative model', 7),
    ('Chapter 4. Mesh | 65\nChapter 4. Mesh | 66', None),
    ('18\nРуководство\nТекст страницы', 18),
])
def test_printed_page_prefers_explicit_running_header(text, expected):
    assert RAGService._printed_page({}, text) == expected


@pytest.mark.parametrize(
    ("product", "file_name", "text", "expected"),
    [
        ("projects", "05_User Manual.pdf", "GeoTechnical analysis System\nwith New eXperience\nUser Manual\nCumulative results.", True),
        ("projects", "GTS_NX_manual.pdf", "Cumulative results.", True),
        ("projects", "manual.pdf", "Generic manual. This example compares GTS NX.", False),
        ("projects", "midas Civil manual.pdf", "Civil manual. Mentions GTS NX.", False),
        ("civil", "GTS_NX_manual.pdf", "GeoTechnical analysis System\nwith New eXperience\nUser Manual", False),
        ("unknown", "manual.pdf", "Unidentified product.", False),
    ],
)
def test_invalid_legacy_tag_requires_positive_product_evidence(
    tmp_path, product, file_name, text, expected
):
    node = NodeWithScore(node=TextNode(text=text, metadata={"product": product, "file_name": file_name}), score=0.9)
    metadata_before = deepcopy(node.node.metadata)
    config = replace(RAGConfig.from_settings(SimpleNamespace()), data_dir=tmp_path, product_scope="strict")
    service = RAGService(config)
    assert service._matches_target_product(node) is expected
    assert node.node.metadata == metadata_before


def test_context_budget_keeps_later_sources_and_redistributes_unused_share():
    texts = ["First sentence. " * 100, "Direct later evidence.", "Third complete sentence. " * 20]
    allocations = RAGService._context_allocations(texts, 180)
    bounded = [RAGService._bounded_context(text, cap) for text, cap in zip(texts, allocations, strict=True)]
    assert sum(allocations) == 180
    assert sum(map(len, bounded)) <= 180
    assert bounded[1] == texts[1]
    assert all(text and text.endswith(".") for text in bounded)
    assert allocations[0] > 60 and allocations[2] > 60


def test_context_budget_never_emits_half_a_structured_row():
    text = 'Column 1=Model | Column 2=Behavior\nColumn 1=A | Column 2=Elastic\nColumn 1=B | Column 2=Elastoplastic'
    cap = text.index("Elastoplastic") + 4
    assert RAGService._bounded_context(text, cap) == text.rsplit("\n", 1)[0]


def test_tiny_context_budget_does_not_cut_a_word():
    assert RAGService._bounded_context("Mesh Generate settings", 12) == "Mesh"


def test_agent_context_keeps_parameters_after_a_lexical_match():
    service = RAGService(RAGConfig.from_settings(SimpleNamespace()))
    text = 'Exact target. ' + 'Intermediate explanatory text. ' * 40 + 'Cohesion c and friction angle phi are required.'
    node = NodeWithScore(node=TextNode(text=text, metadata={'_lexical_focus_query': 'Exact target'}), score=0.9)
    assert 'Cohesion' not in service._context_text(node)
    assert service._context_text(node, focus_lexical=False).endswith('Cohesion c and friction angle phi are required.')


def test_translation_reserves_one_source_without_displacing_primary_evidence(tmp_path):
    service = RAGService(replace(RAGConfig.from_settings(SimpleNamespace()), data_dir=tmp_path, product_scope='strict'))
    def node(identity, text, score):
        return NodeWithScore(node=TextNode(id_=identity, text=text, metadata={'product':'gts_nx', 'file_name': identity+'.pdf'}), score=score)
    primary = [node(str(i), 'Критерии сходимости по усилию перемещению и энергии. Условие '+str(i), .85-i*.01) for i in range(4)]
    english = [node('en'+str(i), 'Convergence force displacement energy tolerance. Variant '+str(i), .9-i*.01) for i in range(3)]
    selection = service._select_context(
        question='Критерии сходимости', plan=RetrievalPlan('Критерии сходимости', ('Критерии сходимости',), 'caller_standalone'),
        rankings=[primary,english], lexical_candidates=[], lexical_query=None, overview_question=False,
        translated_question='Convergence force displacement energy tolerance', translated_candidates=english,
        preserve_full_context=True,
    )
    assert selection.sources[0]['id']=='en0'
    assert any(source['id']=='0' for source in selection.sources)


TABLE_HTML = b'''<table><tr><td><p>Definition before the table.</p>
<table><tr><th>Family</th><th>Model</th><th>Optional</th><th>Behavior</th></tr>
<tr><td rowspan="2">Soil</td><td>Alpha</td><td></td><td>Nonlinear elastic</td></tr>
<tr><td>Beta</td><td>O</td><td>Elastoplastic</td></tr>
<tr><td>Rock</td><td>Gamma</td><td>O</td><td>Elastoplastic</td></tr>
</table><p>Prose after the table.</p></td></tr></table>'''


def test_chm_rows_preserve_behavior_empty_cells_and_rowspans():
    from bs4 import BeautifulSoup

    original = " ".join(BeautifulSoup(TABLE_HTML, "html.parser").get_text(" ", strip=True).split())
    projected = project_table_rows(original, TABLE_HTML)
    assert projected.startswith("Definition before the table.")
    assert projected.endswith("Prose after the table.")
    assert "Column 1=Soil | Column 2=Alpha | Column 3=(empty) | Column 4=Nonlinear elastic" in projected
    assert "Column 1=Soil | Column 2=Beta | Column 3=O | Column 4=Elastoplastic" in projected


def test_chm_partial_row_is_excluded_without_adding_neighboring_facts():
    original = "Definition before the table. Family Model Optional Behavior Soil Alpha Nonlinear elastic Beta O Elastoplastic Rock Gamma O"
    projected = project_table_rows(original, TABLE_HTML)
    assert "Column 2=Alpha" in projected and "Column 2=Beta" in projected
    assert "Gamma" not in projected and "Rock" not in projected
    assert "Prose after" not in projected


def test_chm_chunk_without_header_does_not_guess_column_meanings():
    assert project_table_rows("Beta O Elastoplastic", TABLE_HTML) == ""


def test_chm_unavailable_archive_cannot_expose_flattened_table(tmp_path):
    assert chm_context(tmp_path, "missing.chm", "/member.htm", "Alpha Nonlinear elastic Beta O") == ""


def test_archive_discovery_is_scoped_and_duplicate_names_fail_closed(tmp_path, monkeypatch):
    from app.services import evidence_text

    (tmp_path / "manual.chm").write_bytes(b"first")
    reads = []
    monkeypatch.setattr(evidence_text, "_read_member", lambda *args: reads.append(args) or b"<p>Prose.</p>")
    scans = []
    original = type(tmp_path).rglob
    def scan(path, pattern):
        scans.append(path)
        return original(path, pattern)
    monkeypatch.setattr(type(tmp_path), "rglob", scan)
    with evidence_text.evidence_session(tmp_path):
        assert chm_context(tmp_path, "manual.chm", "/a", "Prose.") == "Prose."
        assert chm_context(tmp_path, "manual.chm", "/b", "Prose.") == "Prose."
    assert len(scans) == 1
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "manual.chm").write_bytes(b"second")
    with evidence_text.evidence_session(tmp_path):
        assert chm_context(tmp_path, "manual.chm", "/a", "Prose.") == ""
    assert len(reads) == 2


def test_parsed_page_reuse_does_not_mutate_other_chunk_provenance():
    from app.services.evidence_text import _parsed_page
    _parsed_page.cache_clear()
    project_table_rows.cache_clear()
    full = "Definition before the table. Family Model Optional Behavior Soil Alpha Nonlinear elastic Beta O Elastoplastic Rock Gamma O Elastoplastic Prose after the table."
    first = project_table_rows(full, TABLE_HTML)
    assert project_table_rows("Beta O Elastoplastic", TABLE_HTML) == ""
    assert _parsed_page.cache_info().misses == 1
    assert _parsed_page.cache_info().hits == 1
    project_table_rows.cache_clear()
    assert project_table_rows(full, TABLE_HTML) == first


def test_duplicate_prefilter_preserves_full_sequence_matcher_decisions():
    import random
    from difflib import SequenceMatcher

    from app.services.rag import NEAR_DUPLICATE_RATIO
    randomizer = random.Random(42)
    for _ in range(80):
        left = " ".join(randomizer.choices(["soil", "load", "model", "stiffness", "stage"], k=80))
        right = left[:180] + " ".join(randomizer.choices(["soil", "load", "model", "stress", "boundary"], k=randomizer.randrange(1, 80)))
        expected = SequenceMatcher(None, left, right, autojunk=False).ratio() >= NEAR_DUPLICATE_RATIO
        assert RAGService._near_duplicate(left, right) == expected


def test_chm_no_table_preserves_prose():
    assert project_table_rows("Useful prose.", b"<p>Useful prose.</p>") == "Useful prose."


def test_table_dom_offsets_do_not_rewrite_identical_preceding_prose():
    html = b'<p>Model Behavior Alpha Elastic</p><p>Actual table:</p><table><tr><th>Model</th><th>Behavior</th></tr><tr><td>Alpha</td><td>Elastic</td></tr></table>'
    chunk = "Model Behavior Alpha Elastic Actual table: Model Behavior Alpha Elastic"
    projected = project_table_rows(chunk, html)
    assert projected.startswith("Model Behavior Alpha Elastic Actual table:")
    assert projected.count("Column 1=Alpha | Column 2=Elastic") == 1
    assert projected.endswith("Column 1=Alpha | Column 2=Elastic")


def test_chm_projection_uses_same_cp1251_decoding_as_ingestion():
    html = '<meta charset="windows-1251"><table><tr><td>Модель</td><td>Поведение</td></tr><tr><td>А</td><td>Упругое</td></tr></table>'.encode("cp1251")
    assert "Column 1=А | Column 2=Упругое" in project_table_rows("Модель Поведение А Упругое", html)


@pytest.mark.parametrize("survivor_score", [0.1, 0.9])
def test_unavailable_chm_cannot_pass_confidence_gate_or_consume_context_slots(tmp_path, survivor_score):
    config = replace(RAGConfig.from_settings(SimpleNamespace()), data_dir=tmp_path, min_score=0.45)
    service = RAGService(config)
    missing = [
        NodeWithScore(node=TextNode(text=f"Unavailable grid {i}", metadata={"file_name": "missing.chm", "chm_member": f"/{i}.htm"}), score=0.95)
        for i in range(5)
    ]
    usable = NodeWithScore(node=TextNode(text="Useful surviving sentence.", metadata={"file_name": "manual.pdf"}), score=survivor_score)
    selection = service._select_context(
        question="question", plan=RetrievalPlan("question", ("question",), "independent"),
        rankings=[[*missing, usable]], lexical_candidates=[], lexical_query=None, overview_question=False,
    )
    assert selection.top_score == survivor_score
    assert selection.has_evidence is (survivor_score >= config.min_score)
    assert selection.retrieved_contexts == ["Useful surviving sentence."]
