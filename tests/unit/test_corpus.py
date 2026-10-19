from __future__ import annotations

import json
from pathlib import Path

import pytest
from llama_index.core import Document
from llama_index.core.embeddings import MockEmbedding
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
)
from qdrant_client import QdrantClient

from app.services.chm_reader import CHMReader
from app.services.corpus import (
    CorpusCatalog,
    apply_reviewed_transcriptions,
    audit_catalog,
    file_hash,
)
from app.services.corpus_html import parse_html
from app.services.ingestion import IngestionConfig, IngestionService, _load_one_file
from app.services.structured_readers import EvidenceSplitter
from scripts.audit_corpus import parse_args, run

HTML = b'''<html lang="en"><title>Literal guide</title><body>
<h1 id="root">Materials</h1><h2 id="soil">Soil</h2>
<p>PRIVATE_MANUAL_TEXT <a href="#soil">Soil details</a></p>
<table aria-describedby="note"><caption>Table 1: parameters</caption>
<tr><th>Name</th><th>Value [kPa]</th></tr>
<tr><td rowspan="2">E</td><td>10</td></tr><tr><td>20</td></tr><tr><td>G</td><td>30</td></tr></table>
<p id="note">Note: drained only.</p><dl><dt>E</dt><dd>Young modulus</dd></dl>
<math>E=10</math><figure><img src="formula.png" alt="Equation 1"><figcaption>Equation 1</figcaption></figure>
</body></html>'''


def write_pdf(path: Path) -> None:
    writer = PdfWriter()
    for number in range(3):
        page = writer.add_blank_page(width=300, height=300)
        if number == 1:
            continue
        font = writer._add_object(DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                                                   NameObject('/Subtype'): NameObject('/Type1'),
                                                   NameObject('/BaseFont'): NameObject('/Helvetica')}))
        resources = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
        content = DecodedStreamObject()
        commands = b'BT /F1 12 Tf 50 250 Td (E = 10 kPa) Tj ET'
        if number == 2:
            image = DecodedStreamObject()
            image.set_data(b'\xff\xff\xff')
            image.update({NameObject('/Type'): NameObject('/XObject'), NameObject('/Subtype'): NameObject('/Image'),
                          NameObject('/Width'): NumberObject(1), NameObject('/Height'): NumberObject(1),
                          NameObject('/ColorSpace'): NameObject('/DeviceRGB'), NameObject('/BitsPerComponent'): NumberObject(8)})
            resources[NameObject('/XObject')] = DictionaryObject({NameObject('/I1'): writer._add_object(image)})
            commands += b' q 40 0 0 30 20 50 cm /I1 Do Q'
        content.set_data(commands)
        page[NameObject('/Resources')] = resources
        page[NameObject('/Contents')] = writer._add_object(content)
    writer.add_outline_item('Explicit section', 0)
    with path.open('wb') as stream:
        writer.write(stream)


def test_common_html_preserves_explicit_structure_without_header_guessing() -> None:
    parsed = parse_html(HTML, 'location')
    blocks = parsed['blocks']
    assert parsed['language'] == 'en'
    assert parsed['title'] == 'Literal guide'
    rows = [b for b in blocks if b['kind'] == 'table_row']
    assert len(rows) == 3
    assert rows[0]['text'] == 'Column 1=E | Column 2=10'
    assert rows[1]['text'] == 'Column 1=E | Column 2=20'
    assert rows[2]['text'] == 'Column 1=G | Column 2=30'
    assert all(any(r['kind'] == 'table_header' for r in row['relations']) for row in rows)
    assert any(r['kind'] == 'defines' for r in parsed['relations'])
    assert any(r['kind'] == 'table_note_for' for r in parsed['relations'])
    formula = next(b for b in blocks if b['kind'] == 'formula')
    assert formula['text'] == 'E=10'
    assert formula['attributes']['verified'] is False
    assert formula['section_path'] == ['Materials', 'Soil']
    image = next(b for b in blocks if b['kind'] == 'image')
    assert image['attributes']['interpretation'] == 'not_performed'
    unlabelled = parse_html(b'<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>', 'other')
    assert not any(b['kind'] == 'table_header' for b in unlabelled['blocks'])


def test_pdf_keeps_blank_pages_glyphs_and_unextracted_images(tmp_path: Path) -> None:
    pdf = tmp_path / 'manual.pdf'
    write_pdf(pdf)
    documents = _load_one_file(pdf, data_root=tmp_path)
    assert [d.metadata['page'] for d in documents] == [1, 2, 3]
    assert [d.metadata['extraction_status'] for d in documents] == ['text', 'no_text', 'text']
    assert all(d.metadata['total_pages'] == 3 for d in documents)
    assert documents[0].metadata['section_path'] == ['Explicit section']
    formula = next(b for b in documents[0].metadata['evidence_blocks'] if b['kind'] == 'formula_candidate')
    assert formula['bbox']
    assert formula['attributes']['glyphs']
    assert formula['attributes']['visual_review_required'] is True
    assert formula['attributes']['verified'] is False
    visual = next(b for b in documents[2].metadata['evidence_blocks'] if b['kind'] == 'visual_unextracted')
    assert visual['attributes']['image_sha256']
    assert visual['bbox'] == [20.0, 220.0, 60.0, 250.0]
    assert visual['text'] == ''
    assert visual['attributes']['ocr_performed'] is False
    assert {r['kind'] for r in documents[1].metadata['relations']} == {'next_page', 'previous_page'}


def test_chm_coverage_and_html_format_parity(tmp_path: Path) -> None:
    class Archive:
        def __init__(self, _path: str):
            pass
        def files(self, **_kwargs):
            return [{'name': '/text.htm'}, {'name': '/blank.htm'}, {'name': '/broken.htm'}, {'name': '/figure.png'}]
        def read(self, name: str, **_kwargs):
            if name == '/broken.htm':
                raise OSError('fixture failure')
            return HTML if name == '/text.htm' else b'<html><body></body></html>'
    source = tmp_path / 'manual.chm'
    source.write_bytes(b'fixture')
    documents = CHMReader(archive_factory=Archive).load_data(source, extra_info={'source_id': 'source'})
    assert len(documents) == 3
    assert [d.metadata['extraction_status'] for d in documents] == ['no_text', 'unreadable', 'text']
    text_document = documents[-1]
    parsed = parse_html(HTML, text_document.metadata['location_id'])
    assert text_document.text == parsed['text']
    assert text_document.metadata['evidence_blocks'] == parsed['blocks']
    assert text_document.metadata['source'] == 'manual.chm::text.htm'


def test_structured_rows_are_atomic_with_explicit_header_context() -> None:
    parsed = parse_html(HTML, 'location')
    document = Document(text=parsed['text'], metadata={'evidence_blocks': parsed['blocks'], 'extraction_version': 'corpus-v1'})
    nodes = EvidenceSplitter(chunk_size=8, chunk_overlap=1)([document])
    rows = [node for node in nodes if node.metadata['block_type'] == 'table_row']
    assert len(rows) == 3
    assert all('Column 1=Name | Column 2=Value [kPa]\n' in row.text for row in rows)
    assert all(row.metadata['atomic_evidence'] for row in rows)
    assert all(row.ref_doc_id == document.id_ for row in nodes)
    assert [n.id_ for n in nodes] == [n.id_ for n in EvidenceSplitter(chunk_size=8, chunk_overlap=1)([document])]


def test_chm_unicode_output_fallback_keeps_exact_member_and_checks_size(tmp_path: Path, monkeypatch) -> None:
    from pylibmspack import ChmArchive, _cab

    member = '/PM4Sand — копия.htm'
    source = tmp_path / 'manual.chm'
    source.write_bytes(b'fixture')
    archive = ChmArchive(str(source))
    monkeypatch.setattr(archive, 'files', lambda **kwargs: [{'name': member, 'size': len(HTML)}])
    def failed_read(name, **kwargs):
        assert name == member
        raise FileNotFoundError('native Unicode output path')
    monkeypatch.setattr(archive, 'read', failed_read)
    calls = []
    def exact_extract(path, name, target, max_size):
        assert path == str(source)
        assert name == member
        assert str(target).isascii()
        assert Path(target).name == 'member.bin'
        calls.append(name)
        Path(target).write_bytes(HTML)
        return 0
    monkeypatch.setattr(_cab, 'chm_extract_file', exact_extract)
    # Keep the native output-path compatibility requirement portable in tests.
    monkeypatch.setattr('tempfile.tempdir', str(tmp_path))
    reader = CHMReader(archive_factory=lambda _: archive)
    document = reader.load_data(source)[0]
    assert document.metadata['chm_member'] == member
    assert document.metadata['chm_read_method'] == 'pylibmspack_ascii_output'
    assert document.metadata['extraction_status'] == 'text'
    assert calls == [member]
    assert document.text == parse_html(HTML, document.metadata['location_id'])['text']
    monkeypatch.setattr(archive, 'files', lambda **kwargs: [{'name': member, 'size': len(HTML) + 1}])
    failure = reader.load_data(source)[0]
    assert failure.metadata['extraction_status'] == 'unreadable'
    assert failure.metadata['extraction_error'] == 'ChmError'


def test_catalog_resolves_literal_links_and_keeps_summary_text_free(tmp_path: Path) -> None:
    (tmp_path / 'guide.html').write_bytes(HTML)
    (tmp_path / 'support_documents.json').write_text('synthetic', encoding='utf8')
    first = CorpusCatalog.build(tmp_path)
    second = CorpusCatalog.build(tmp_path)
    assert first.as_dict() == second.as_dict()
    assert first.summary()['sources'] == 1
    assert 'PRIVATE_MANUAL_TEXT' not in json.dumps(first.summary())
    assert any(r['kind'] == 'explicit_link' for r in first.relations)
    assert not any(r['kind'] in {'semantic_similarity', 'continuation'} for r in first.relations)
    saved = tmp_path / 'catalog.json'
    first.persist(saved)
    assert CorpusCatalog.load(saved).as_dict() == first.as_dict()
    (tmp_path / 'guide.html').write_bytes(HTML + b'changed')
    audit = audit_catalog(first, data_root=tmp_path)
    assert audit['sources']['changed_sources'] == 1


def test_visual_transcriptions_require_exact_source_review_and_preserve_image(tmp_path: Path) -> None:
    pdf = tmp_path / 'manual.pdf'
    write_pdf(pdf)
    documents = _load_one_file(pdf, data_root=tmp_path)
    d = documents[2]
    image = next(b for b in d.metadata['evidence_blocks'] if b['kind'] == 'visual_unextracted')
    record = {'source_hash': file_hash(pdf), 'page': 3, 'bbox': image['bbox'],
              'image_sha256': image['attributes']['image_sha256'], 'product': 'gts_nx', 'version': 'unknown',
              'text': 'Verified equation transcription', 'kind': 'formula', 'reviewer': 'Fixture reviewer',
              'reviewed_at': '2026-10-16', 'review_id': 'review-1', 'approval_status': 'pending'}
    before = d.text
    apply_reviewed_transcriptions(documents, [record])
    assert d.text == before
    record['approval_status'] = 'approved'
    with pytest.raises(ValueError, match='exact source'):
        apply_reviewed_transcriptions(documents, [{**record, 'source_hash': 'wrong'}])
    apply_reviewed_transcriptions(documents, [record])
    assert 'Verified equation transcription' in d.text
    assert image in d.metadata['evidence_blocks']
    transcript = d.metadata['evidence_blocks'][-1]
    assert transcript['attributes']['verified'] is True
    assert transcript['relations'] == [{'kind': 'transcription_of', 'to_id': image['block_id']}]


def test_ingestion_sidecar_coverage_idempotency_and_blank_replacement(tmp_path: Path) -> None:
    data = tmp_path / 'data'
    data.mkdir()
    pdf = data / 'manual.pdf'
    write_pdf(pdf)
    service = IngestionService(IngestionConfig(data_dir=data, docstore_path=tmp_path / 'docstore.json',
                                                embedding_dimensions=8, chunk_size=64, chunk_overlap=4),
                               client=QdrantClient(':memory:'), embed_model=MockEmbedding(embed_dim=8))
    first = service.ingest()
    assert first.extracted_locations == 3
    assert first.no_text_locations == 1
    assert first.loaded_documents == 2
    sidecar = tmp_path / 'docstore.corpus.json'
    assert CorpusCatalog.load(sidecar).summary()['locations'] == 3
    assert service.ingest().changed_documents == 0
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with pdf.open('wb') as stream:
        writer.write(stream)
    replaced = service.ingest(pdf)
    assert replaced.loaded_documents == 0
    assert replaced.no_text_locations == 1
    assert service._client.count(service.config.collection_name).count == 0
    service.close()


def test_audit_cli_is_read_only_and_handles_partial_snapshot(tmp_path: Path) -> None:
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'guide.html').write_bytes(HTML)
    snapshot = tmp_path / 'snapshot.json'
    snapshot.write_text(json.dumps({'points': [{'id': '1', 'payload': {'doc_id': 'other'}}]}), encoding='utf8')
    before = sorted(str(p) for p in tmp_path.rglob('*'))
    report = run(parse_args(['--data-dir', str(data), '--qdrant-snapshot', str(snapshot)]))
    assert sorted(str(p) for p in tmp_path.rglob('*')) == before
    assert report['qdrant']['snapshot_complete'] is False
    assert report['qdrant']['conclusive'] is False
    assert report['qdrant']['repair_applied'] is False
    assert 'PRIVATE_MANUAL_TEXT' not in json.dumps(report)


def test_parse_error_preserves_original_and_source_failure_ledger(tmp_path: Path) -> None:
    (tmp_path / 'broken.pdf').write_bytes(b'not pdf')
    catalog = CorpusCatalog.build(tmp_path)
    assert catalog.summary()['failed_sources'] == 1
    assert (tmp_path / 'broken.pdf').exists()
    assert catalog.sources[0]['error_type']


def test_visual_observation_must_match_hash_and_physical_page(tmp_path: Path) -> None:
    pdf = tmp_path / 'manual.pdf'
    write_pdf(pdf)
    catalog = CorpusCatalog.build(tmp_path)
    observation = {'source_hash': file_hash(pdf), 'page': 2, 'status': 'blank_verified',
                   'reviewer': 'Fixture reviewer', 'reason': 'Rendered original page is visually blank'}
    catalog.apply_observations([observation])
    assert catalog.summary()['visual_reviews'] == 1
    with pytest.raises(ValueError):
        catalog.apply_observations([{**observation, 'source_hash': 'stale'}])


def test_pdf_outline_confirms_cross_page_section_without_physical_neighbor_guess(tmp_path: Path) -> None:
    pdf = tmp_path / 'manual.pdf'
    write_pdf(pdf)
    catalog = CorpusCatalog.build(tmp_path)
    heading = next(b for b in catalog.blocks if b['kind'] == 'outline_heading')
    assert heading['text'] == 'Explicit section'
    assert heading['attributes'] == {'method': 'pdf_explicit_outline', 'destination_page': 1}
    third = next(loc for loc in catalog.locations if loc['page'] == 3)
    third_blocks = [b for b in catalog.blocks if b['location_id'] == third['location_id']]
    assert all({'kind': 'parent_section', 'to_id': heading['block_id']} in b['relations'] for b in third_blocks)
    assert not any(r['kind'] == 'continued_table' for r in catalog.relations)


@pytest.mark.parametrize('marker,width,expected', [('Table 7 (continued)', 2, True),
                                                  ('Table 7', 2, False),
                                                  ('Table 7 (continued)', 3, False),
                                                  ('Table 8 (continued)', 2, False)])
def test_table_continuation_requires_number_marker_and_confirmed_width(marker: str, width: int, expected: bool) -> None:
    catalog = CorpusCatalog(sources=[{'source_id': 's', 'format': 'pdf', 'relative_path': 'manual.pdf'}])
    for page, caption, columns in ((1, 'Table 7', 2), (2, marker, width)):
        lid, tid, cid = f'location{page}', f'table{page}', f'caption{page}'
        catalog.locations.append({'source_id': 's', 'location_id': lid, 'page': page, 'section_path': []})
        catalog.blocks.extend([
            {'source_id': 's', 'location_id': lid, 'block_id': tid, 'kind': 'table', 'text': '', 'relations': []},
            {'source_id': 's', 'location_id': lid, 'block_id': cid, 'kind': 'paragraph', 'text': caption, 'relations': []},
            {'source_id': 's', 'location_id': lid, 'block_id': f'row{page}', 'kind': 'table_row', 'text': 'literal row',
             'attributes': {'table_id': tid, 'cells': [{} for _ in range(columns)]}, 'relations': []},
        ])
        catalog.relations.append({'source_id': 's', 'kind': 'table_caption_for', 'from_id': cid, 'to_id': tid})
    catalog._resolve_relations()
    continued = [r for r in catalog.relations if r['kind'] == 'continued_table']
    assert bool(continued) is expected
    if expected:
        assert continued[0]['from_id'] == 'table2'
        assert continued[0]['to_id'] == 'table1'


def test_two_column_pdf_keeps_column_sentences_instead_of_interleaving(tmp_path: Path) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=500, height=400)
    font = writer._add_object(DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                                               NameObject('/Subtype'): NameObject('/Type1'),
                                               NameObject('/BaseFont'): NameObject('/Helvetica')}))
    page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
    commands = [b'BT /F1 12 Tf 140 370 Td (Full width independent heading) Tj ET']
    for row in range(4):
        y = 330 - row * 25
        for x, label in ((20, 'Left'), (290, 'Right')):
            commands.append(f'BT /F1 12 Tf {x} {y} Td ({label} sentence number {row}) Tj ET'.encode('ascii'))
    stream = DecodedStreamObject()
    stream.set_data(b'\n'.join(commands))
    page[NameObject('/Contents')] = writer._add_object(stream)
    pdf = tmp_path / 'columns.pdf'
    with pdf.open('wb') as output:
        writer.write(output)
    documents = _load_one_file(pdf, data_root=tmp_path)
    body = [b for b in documents[0].metadata['evidence_blocks'] if b['text'].startswith(('Left', 'Right'))]
    assert [b['text'] for b in body] == ([f'Left sentence number {i}' for i in range(4)]
                                       + [f'Right sentence number {i}' for i in range(4)])
    assert all(b['attributes']['column_detection'] == 'repeated_geometric_gutter' for b in body)
    assert all(b['attributes']['reading_order'] == 'layout_candidate_unverified' for b in body)
    nodes = EvidenceSplitter(chunk_size=1024, chunk_overlap=0)(documents)
    assert not any('Left sentence' in node.text and 'Right sentence' in node.text for node in nodes)
