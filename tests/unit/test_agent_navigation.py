"""Navigation IDs authorize exact read-only pages within the current turn."""
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.services.agent_persistent import build_agent, search_knowledge_base
from app.services.prepared_evidence import (
    atomic_unit_fingerprint,
    bounded_navigation,
    navigation_reference,
)


def test_navigation_tool_schema_exposes_the_same_bounds_as_the_reader():
    schema = search_knowledge_base.args_schema.model_json_schema()
    array = next(item for item in schema['properties']['navigation_ids']['anyOf'] if item['type'] == 'array')
    assert array['minItems'] == 1 and array['maxItems'] == 3


def test_short_references_are_stable_when_followup_reorders_the_manifest():
    items = [{'id': identity, 'hint': identity[-4:], 'not_citable': True} for identity in ('a' * 64, 'b' * 64)]
    raw, projected, truncated = bounded_navigation(items)
    assert not truncated and [row['id'] for row in raw] == ['a' * 64, 'b' * 64]
    assert all(len(row['id']) == 12 for row in projected)
    _, reordered, _ = bounded_navigation(list(reversed(items)))
    assert [row['id'] for row in reordered] == [row['id'] for row in reversed(projected)]


def test_reference_collisions_fail_closed_even_when_second_item_follows_the_cap(monkeypatch):
    identities = ['first', *[f'unique-{i}' for i in range(110)], 'last']
    def reference(identity):
        return 'collision' if identity in {'first', 'last'} else identity
    monkeypatch.setattr('app.services.prepared_evidence.navigation_reference', reference)
    raw, visible, truncated = bounded_navigation([{'id': identity, 'hint': 'literal', 'not_citable': True}
                                                for identity in identities])
    assert truncated and len(raw) <= 100
    assert not ({'first', 'last'} & {row['id'] for row in raw})
    assert 'collision' not in {row['id'] for row in visible}


class RAG:
    supports_prepared_evidence = True
    config = SimpleNamespace(request_timeout=2)
    def __init__(self):
        self.reads = []
    def retrieve(self, query, **kwargs):
        return {'status': 'ok', 'sources': [{'chunk_id': 'a', 'file_name': 'manual.pdf'}],
                'contexts': ['Main documented option.'], '_navigation': [
                    {'id': 'server-id', 'page': 3, 'file_name': 'manual.pdf',
                     'hint': 'Additional option', 'not_citable': True,
                     'metadata': {'source_hash': 'server-only-binding'}}]}
    def read_navigation(self, items, **kwargs):
        self.reads.append(items)
        return {'status': 'ok', 'sources': [{'chunk_id': 'b', 'file_name': 'manual.pdf',
                                          'page': 3, 'atomic_evidence': True}],
                'contexts': ['Additional option is available under condition B.']}


class Model:
    def __init__(self, ids):
        self.ids, self.packets, self.count = ids, [], 0
    async def ainvoke(self, messages, **kwargs):
        self.count += 1
        self.packets.append('\n'.join(str(message.content) for message in messages))
        if self.count == 1:
            return AIMessage(content='', tool_calls=[{'id': 'read', 'name': 'search_knowledge_base',
                'args': {'query': 'Прочитай дополнительный вариант', 'english_query': 'Read additional option',
                         'navigation_ids': self.ids}}])
        return AIMessage(content='Main option [1]. Additional option under condition B [2].')


@pytest.mark.asyncio
async def test_navigation_reads_exact_server_binding_and_merges_stable_citations(monkeypatch):
    monkeypatch.setattr('app.chat.service._o200k_base_encoding', lambda: None)
    rag, model = RAG(), Model([navigation_reference('server-id')])
    graph = build_agent(InMemorySaver(), model=model, rag_provider=lambda: rag)
    state = await graph.ainvoke({'messages': [HumanMessage(content='Какие варианты доступны?')]},
                               {'configurable': {'thread_id': 'navigation'}})
    assert rag.reads[0][0]['metadata']['source_hash'] == 'server-only-binding'
    assert 'server-only-binding' not in model.packets[0]
    assert 'server-id' not in model.packets[0] and navigation_reference('server-id') in model.packets[0]
    assert rag.reads[0][0]['id'] == 'server-id' and 'not_citable' in model.packets[0]
    assert model.packets[1].count('"hint": "Additional option"') == 1
    assert [record['citation_number'] for record in state['citation_evidence']] == [1, 2]
    assert state['citation_evidence'][1]['text'] == 'Additional option is available under condition B.'
    assert state['prepared_evidence']['latest_evidence_change'] == {
        'new_records': 1, 'added_text_chars': len('Additional option is available under condition B.'),
        'semantic_coverage': None}


@pytest.mark.asyncio
@pytest.mark.parametrize('ids', [['other-id'], ['server-id'], [navigation_reference('server-id')] * 2, [], 'server-id'])
async def test_unsupplied_or_invalid_navigation_ids_cannot_read_documents(monkeypatch, ids):
    monkeypatch.setattr('app.chat.service._o200k_base_encoding', lambda: None)
    rag, model = RAG(), Model(ids)
    graph = build_agent(InMemorySaver(), model=model, rag_provider=lambda: rag)
    state = await graph.ainvoke({'messages': [HumanMessage(content='Какие варианты доступны?')]},
                               {'configurable': {'thread_id': 'invalid-navigation'}})
    assert rag.reads == [] and len(state['citation_evidence']) == 1
    assert 'navigation_ids' in json.dumps(state['tool_results'], ensure_ascii=False)


@pytest.mark.asyncio
async def test_reference_from_previous_turn_cannot_read_a_new_turn(monkeypatch):
    monkeypatch.setattr('app.chat.service._o200k_base_encoding', lambda: None)
    class NextTurnRAG(RAG):
        def __init__(self):
            super().__init__()
            self.searches = 0
        def retrieve(self, query, **kwargs):
            self.searches += 1
            result = super().retrieve(query, **kwargs)
            if self.searches > 1:
                result['_navigation'] = []
            return result
    class NextTurnModel(Model):
        async def ainvoke(self, messages, **kwargs):
            self.count += 1
            if self.count == 2:
                return AIMessage(content='', tool_calls=[{'id': 'stale-read', 'name': 'search_knowledge_base',
                    'args': {'query': 'Прочитай прежний раздел', 'english_query': 'Read previous section',
                             'navigation_ids': [navigation_reference('server-id')]}}])
            return AIMessage(content='Main documented option [1].')
    rag, model = NextTurnRAG(), NextTurnModel([])
    graph = build_agent(InMemorySaver(), model=model, rag_provider=lambda: rag)
    config = {'configurable': {'thread_id': 'next-turn-navigation'}}
    await graph.ainvoke({'messages': [HumanMessage(content='Какие варианты доступны?')]}, config)
    state = await graph.ainvoke({'messages': [HumanMessage(content='Как изменить настройку?')]}, config)
    assert state['evidence_navigation'] == [] and rag.reads == []
    result = json.loads(state['tool_results'][-1]['result'])
    assert result['status'] == 'error' and 'navigation_ids' in result['message']


@pytest.mark.asyncio
async def test_followup_packing_receives_only_already_visible_whole_units(monkeypatch):
    monkeypatch.setattr('app.chat.service._o200k_base_encoding', lambda: None)
    source = {'chunk_id': 'whole', 'block_id': 'whole', 'location_id': 'section',
              'source_id': 'manual', 'source_hash': 'current-bytes', 'product': 'gts_nx',
              'version': 'unknown', 'atomic_evidence': True, 'file_name': 'manual.pdf'}
    class WholeRAG(RAG):
        def __init__(self):
            super().__init__()
            self.options = []
        def retrieve(self, query, **kwargs):
            self.options.append(kwargs)
            if len(self.options) == 1:
                return {'status': 'ok', 'sources': [source], 'contexts': ['Whole accepted section.']}
            return {'status': 'ok', 'sources': [{'chunk_id': 'new', 'file_name': 'manual.pdf'}],
                    'contexts': ['New detail.']}
    class SearchModel(Model):
        async def ainvoke(self, messages, **kwargs):
            self.count += 1
            if self.count == 1:
                return AIMessage(content='', tool_calls=[{'id': 'more', 'name': 'search_knowledge_base',
                    'args': {'query': 'Найди условие', 'english_query': 'Find condition'}}])
            return AIMessage(content='Whole accepted section [1]. New detail [2].')
    rag, model = WholeRAG(), SearchModel([])
    graph = build_agent(InMemorySaver(), model=model, rag_provider=lambda: rag)
    state = await graph.ainvoke({'messages': [HumanMessage(content='Какое условие нужно?')]},
                               {'configurable': {'thread_id': 'known-whole-units'}})
    assert rag.options[1]['known_atomic_units'] == frozenset({atomic_unit_fingerprint(source, 'Whole accepted section.')})
    assert [record['text'] for record in state['citation_evidence']] == ['Whole accepted section.', 'New detail.']
    assert [record['citation_number'] for record in state['citation_evidence']] == [1, 2]


@pytest.mark.asyncio
async def test_navigation_discovered_by_followup_is_visible_and_readable(monkeypatch):
    monkeypatch.setattr('app.chat.service._o200k_base_encoding', lambda: None)
    class FollowupRAG(RAG):
        def __init__(self):
            super().__init__()
            self.searches = 0
        def retrieve(self, query, **kwargs):
            self.searches += 1
            result = super().retrieve(query, **kwargs)
            if self.searches == 1:
                result.pop('_navigation')
            return result
    class FollowupModel(Model):
        async def ainvoke(self, messages, **kwargs):
            current = self.count
            self.count += 1
            if current == 0:
                return AIMessage(content='', tool_calls=[{'id': 'search', 'name': 'search_knowledge_base',
                    'args': {'query': 'Уточни варианты', 'english_query': 'Find further options'}}])
            if current == 1:
                payload = '\n'.join(str(message.content) for message in messages)
                assert navigation_reference('server-id') in payload and 'server-only-binding' not in payload
                return AIMessage(content='', tool_calls=[{'id': 'read', 'name': 'search_knowledge_base',
                    'args': {'query': 'Прочитай страницу', 'english_query': 'Read page', 'navigation_ids': [navigation_reference('server-id')]}}])
            return AIMessage(content='Main option [1]. Additional option under condition B [2].')
    rag, model = FollowupRAG(), FollowupModel(['server-id'])
    graph = build_agent(InMemorySaver(), model=model, rag_provider=lambda: rag)
    state = await graph.ainvoke({'messages': [HumanMessage(content='Какие варианты доступны?')]},
                               {'configurable': {'thread_id': 'followup-navigation'}})
    assert len(rag.reads) == 1
    assert state['prepared_evidence']['navigation'][0]['id'] == navigation_reference('server-id')
    assert state['citation_evidence'][1]['source']['page'] == 3
