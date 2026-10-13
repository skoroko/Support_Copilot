import pytest

from bot.text_chunks import split_text


@pytest.mark.parametrize("text", ["", "Короткий ответ.", "Абзац.\n\n" * 700, "😀" * 4001],
                         ids=["empty", "short", "paragraphs", "astral"])
def test_chunks_preserve_every_character_and_utf16_safety(text):
    chunks = split_text(text)
    assert "".join(chunks) == text
    assert all(len(chunk) <= 1800 for chunk in chunks)
    assert all(len(chunk.encode("utf-16-le")) // 2 <= 3600 for chunk in chunks)


def test_parameter_heading_stays_with_body_when_the_pair_fits():
    intro = "Вводный абзац. " * 110 + "\n\n"
    pair = "### E50ref\n\nЭто секущая жёсткость [1].\n\n"
    text = intro + pair + "Ещё один абзац. " * 110
    chunks = split_text(text)
    assert "".join(chunks) == text
    assert any(pair in chunk for chunk in chunks)


def test_splitter_avoids_breaking_bold_label_near_limit():
    text = "Текст. " * 255 + "**E50ref** — описание параметра.\n" + "Другой текст. " * 30
    chunks = split_text(text)
    assert "".join(chunks) == text
    assert any("**E50ref**" in chunk for chunk in chunks)
    assert all(chunk.count("**") % 2 == 0 for chunk in chunks)


def test_hard_fallback_splits_overlong_indivisible_unicode_line_losslessly():
    text = "😀Ж" * 2000
    assert split_text(text) == [text[:1800], text[1800:3600], text[3600:]]


def test_invalid_chunk_limit_is_rejected():
    with pytest.raises(ValueError):
        split_text("text", 0)
