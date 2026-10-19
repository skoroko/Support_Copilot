import pytest

from app.services.answer_quality import (
    ANSWER_QUALITY_UNAVAILABLE,
    answer_quality_issue,
    compact_answer,
)


@pytest.mark.parametrize("text", ["Как это понимать практичесki", "Сделайте пpoверку.", "### Параметpы модели"])
def test_mixed_alphabets_in_prose_are_detected(text):
    assert answer_quality_issue(text, "stop") == "mixed_script"


@pytest.mark.parametrize(
    "text",
    [
        "Параметры Hardening Soil: E50ref, Eoedref, Eurref, G0ref, γ0.7, c, φ, ψ [1].",
        "Откройте GTS_NX 2021.chm и файл Параметpы.pdf.",
        "Код: `практичесki = Eurref`.\n```python\nпpoверка = 1\n```",
        "Ссылка: https://example.org/практичесki и адрес пpoверка@example.org.",
        "Полностью кириллический законченный ответ.",
    ],
)
def test_technical_literals_and_valid_russian_are_preserved(text):
    assert answer_quality_issue(text, "stop") is None


def test_provider_length_stop_is_detected_even_if_text_looks_complete():
    assert answer_quality_issue("Короткий ответ.", "length") == "length"
    assert answer_quality_issue(" ", "stop") == "empty"


def test_compaction_preserves_literals_citations_and_complete_bullets():
    text = "- Параметр `phi_А` задаёт угол φ [2].\n- См. https://example.org/тестA [1].\n- **c, φ, ψ"
    compact, incomplete = compact_answer(text, "length")
    assert compact == "\n".join(text.splitlines()[:2])
    assert incomplete is True


def test_compaction_preserves_multiline_code_and_citation_after_period():
    text = "```python\nphi_А = 30\nprint(phi_А)\n```\nЗавершено. [1] Следующий оборванный"
    assert compact_answer(text, "length") == (text.split(" Следующий")[0], True)
    assert compact_answer("См. https://example.org/параметрA.\nОборвано", "length") == (
        "См. https://example.org/параметрA.", True)


def test_compaction_drops_only_defective_sentence():
    assert compact_answer("Первое предложение. Это практичесki. Последнее предложение.") == (
        "Первое предложение. Последнее предложение.", True)
    assert compact_answer("", "stop") == (ANSWER_QUALITY_UNAVAILABLE, True)


def test_compaction_keeps_parameter_labels_with_complete_bodies_and_drops_truncated_tail():
    complete = (
        "### E50ref\n\nЭто секущая жёсткость при 50% прочности [1].\n"
        "2. **Eoedref**\nЭто жёсткость при одномерном сжатии [2].\n"
        "**Eurref**\nЭто жёсткость при разгрузке и повторном нагружении [3].\n"
        "`m`\nСтепень зависимости жёсткости от напряжения: m = 0.5 [1]."
    )
    text = complete + "\n### Диапазон m\n**0.5 ≤ m ≤ 1"
    compact, incomplete = compact_answer(text, "length")
    assert compact == complete.replace("\n\n", "\n")
    assert incomplete is True
    assert "Диапазон" not in compact
    assert "0.5 ≤" not in compact


@pytest.mark.parametrize("heading", ["### E50ref", "2. **Eoedref**", "**Eurref**", "`m`"])
def test_compaction_drops_label_without_complete_following_body(heading):
    assert compact_answer(heading + "\nЭто секущая", "length") == (ANSWER_QUALITY_UNAVAILABLE, True)


def test_compaction_preserves_numbered_complete_parameter_item_and_decimal_formula():
    text = "1. **E50ref** — жёсткость при ε = 0.5 [1].\n2. **Eoedref** — оборвано"
    assert compact_answer(text, "length") == (text.splitlines()[0], True)
    assert compact_answer("1.\n**Eurref**\nОборвано", "length") == (ANSWER_QUALITY_UNAVAILABLE, True)


def test_compaction_retains_heading_for_complete_code_or_table_body():
    text = "### Формула\n```text\nE = 0.5\n```\n### Таблица\n| m | 0.5 [1] |\n### Оборвано"
    assert compact_answer(text, "length") == (text.rsplit("\n", 1)[0], True)


def test_orphan_label_is_not_attached_to_next_parameter_description():
    text = "### 1. E50ref\n### 2. Eoedref\nЭто жёсткость при сжатии [1].\n### 3. Eurref\nОборвано"
    assert compact_answer(text, "length") == ("### 2. Eoedref\nЭто жёсткость при сжатии [1].", True)


def test_compaction_treats_formatting_inside_code_as_opaque():
    text = "### Формула\n```python\nvalue = 2 ** 3\n```\nКод: `value = 2 ** 3`.\nОборвано"
    assert compact_answer(text, "length") == (text.rsplit("\n", 1)[0], True)


@pytest.mark.parametrize(
    "text",
    [
        "Опора воспринимает нагрузку 120 kПа [1].",
        "Жёсткость пружины равна 50 kН/м на сваю.",
        "Напряжение достигает 0.5 MПa на срезе элемента.",
        "Модуль реакции задаётся в kPa и кН/м³.",
        "Формула Fтр = kN задаёт трение [2].",
        "Единицы: kВт, MПa, kДж, мkF.",
    ],
)
def test_technical_mixed_units_are_preserved(text):
    assert answer_quality_issue(text, "stop") is None


def test_compaction_keeps_mixed_unit_sentence():
    compact, incomplete = compact_answer("Нагрузка 120 kПа передаётся на сваи. Это пpoверка.")
    assert compact == "Нагрузка 120 kПа передаётся на сваи."
    assert incomplete is True


def test_long_mojibake_word_is_still_detected():
    assert answer_quality_issue("Проверьте парамeтры модели.", "stop") == "mixed_script"
