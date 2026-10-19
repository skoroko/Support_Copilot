from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.corpus import CorpusCatalog, apply_reviewed_transcriptions
from app.services.ingestion import _load_one_file
from app.services.visual_recovery import (
    build_visual_inventory,
    make_visual_contact_sheets,
    render_review_packet,
    render_unique_visual_batch,
    select_dev_pages,
    verify_candidate_locators,
)
from tests.unit.test_corpus import write_pdf


def test_original_inventory_packet_pending_and_approved_ingest(tmp_path: Path) -> None:
    data = tmp_path / 'data'
    data.mkdir()
    source = data / 'manual.pdf'
    write_pdf(source)
    catalog = CorpusCatalog.build(data)
    catalog_path = tmp_path / 'catalog.json'
    catalog.persist(catalog_path)
    inventory = build_visual_inventory(catalog_path, data)
    assert inventory['summary']['regions'] == 1
    assert inventory['summary']['important_text_bearing_total'] is None
    region = inventory['regions'][0]
    assert region['source_pixel_size'] == [1, 1]
    assert region['semantic_class'] == 'unreviewed'
    assert region['bbox'] == [20.0, 220.0, 60.0, 250.0]
    packet = render_review_packet(inventory, data, [{'source_hash': region['source_hash'], 'page': 3}], tmp_path / 'review')
    assert packet['approval_performed'] is False
    candidate = json.loads((tmp_path / 'review/candidates.json').read_text())[0]
    assert Path(candidate['crop_path']).is_file()
    assert Path(candidate['page_png']).is_file()
    assert candidate['image_sha256'] == region['image_sha256']
    batch = render_unique_visual_batch(inventory, data, tmp_path / 'batch')
    assert batch['complete'] is True
    assert batch['prepared_groups'] == batch['prepared_occurrences'] == 1
    group = batch['groups'][0]
    assert group['occurrences'][0]['region_id'] == region['region_id']
    assert Path(group['full_crop_path']).is_file()
    assert Path(group['classifier_crop_path']).is_file()
    assert batch['provider_calls'] == 0
    sheets = make_visual_contact_sheets(tmp_path / 'batch/batch_manifest.json', tmp_path / 'sheets')
    assert sheets['tile_count'] == 1
    tile = sheets['sheets'][0]['tiles'][0]
    assert tile['tile_id'] == 'V00001'
    assert tile['group_id'] == group['group_id']
    assert Path(sheets['sheets'][0]['path']).is_file()
    assert verify_candidate_locators(inventory, [candidate])['all_locators_match'] is True
    altered = dict(candidate, image_sha256='not original')
    assert verify_candidate_locators(inventory, [altered])['all_locators_match'] is False
    documents = _load_one_file(source, data_root=data)
    prior = documents[2].text
    candidate['text'] = 'independently reviewed white square'
    apply_reviewed_transcriptions(documents, [candidate])
    assert documents[2].text == prior  # pending candidates are never evidence
    candidate.update(approval_status='approved', reviewer='independent fixture', reviewed_at='2026-10-16', review_id='review-fixture')
    apply_reviewed_transcriptions(documents, [candidate])
    assert candidate['text'] in documents[2].text
    assert any(b['attributes'].get('method') == 'reviewed_visual_transcription' for b in documents[2].metadata['evidence_blocks'])
    transcript = documents[2].metadata['evidence_blocks'][-1]
    assert transcript['attributes']['verification_scope'] == 'literal_image_transcription'
    assert transcript['attributes']['semantic_verified'] is False
    source.write_bytes(source.read_bytes() + b'\nchanged')
    with pytest.raises(ValueError, match='changed'):
        build_visual_inventory(catalog_path, data)
    with pytest.raises(ValueError, match='unchanged'):
        render_review_packet(inventory, data, [{'source_hash': region['source_hash'], 'page': 3}], tmp_path / 'bad')


def test_dev_selection_refuses_holdout_and_covers_other_sources(tmp_path: Path) -> None:
    dev_path = tmp_path / 'dev_cases.json'
    dev_path.write_text(json.dumps([{'id': 'dev-1', 'split': 'dev', 'required_evidence_slots': [
        {'file_sha256': 'source-a', 'physical_page': 2}]}]))
    inventory = {'regions': [
        {'source_hash': 'source-a', 'page': 2, 'triage_class': 'wide_strip_candidate'},
        {'source_hash': 'source-b', 'page': 1, 'triage_class': 'diagram_screenshot_or_other_candidate'},
        {'source_hash': 'source-b', 'page': 3, 'triage_class': 'wide_strip_candidate'},
    ]}
    selection = select_dev_pages(inventory, dev_path, ['dev-1'])
    assert [(s['source_hash'], s['page']) for s in selection] == [('source-a', 2), ('source-b', 3)]
    with pytest.raises(ValueError, match='Holdout'):
        select_dev_pages(inventory, tmp_path / 'holdout_cases.json', ['dev-1'])
    with pytest.raises(ValueError, match='explicit dev'):
        select_dev_pages(inventory, dev_path, ['absent'])
